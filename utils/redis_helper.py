import uuid
import math
import time
import redis
import msgpack

REDIS_HOST='172.30.35.11'
REDIS_PORT=6379
REDIS_PASS="iflytekq1w2E#R$"

RECONSTRUCT_QUEUE="reconstruct_queue"
RECONSTRUCT_PRIORITY_QUEUE="reconstruct_priority_queue"
RECONSTRUCT_LOCK_KEY="recon_lock"

MULTICLIPS_QUEUE="multiclips_queue"
MULTICLIPS_PRIORITY_QUEUE="multiclips_priority_queue"
MULTICLIPS_LOCK_KEY="multiclips_lock"

HPP_QUEUE="hpp_queue"
HPP_PRIORITY_QUEUE="hpp_priority_queue"
HPP_LOCK_KEY="hpp_lock"

def get_redis_rcon():
    pool = redis.ConnectionPool(host='172.30.35.11',
                                port=6379, db=1, password="iflytekq1w2E#R$")
    return redis.StrictRedis(connection_pool=pool)

class Message(object):
    def __init__(self, message_type, data, node_id):
        self.type = message_type
        self.data = data
        self.node_id = node_id

    def serialize(self):
        return msgpack.dumps((self.type, self.data, self.node_id))

    @classmethod
    def unserialize(cls, data):
        msg = cls(*msgpack.loads(data))
        #msg = cls(*msgpack.loads(data, encoding='utf-8'))
        return msg  
    
def push_msg(rcon, queue, task):
    repeat_msg = Message("json", task, None)
    result = rcon.rpush(queue, repeat_msg.serialize())
    return result

def read_msg(rcon, queue):
    item = rcon.blpop(queue, 0)[1]            
    msg = Message.unserialize(item) 
    return msg.data

def acquire_lock_with_timeout(conn, lock_name, acquire_timeout=3, lock_timeout=2):
    """
    基于 Redis 实现的分布式锁
    
    :param conn: Redis 连接
    :param lock_name: 锁的名称
    :param acquire_timeout: 获取锁的超时时间，默认 3 秒
    :param lock_timeout: 锁的超时时间，默认 2 秒
    :return:
    """

    identifier = str(uuid.uuid4())
    lockname = f'lock:{lock_name}'
    lock_timeout = int(math.ceil(lock_timeout))

    end = time.time() + acquire_timeout

    while time.time() < end:
        # 如果不存在这个锁则加锁并设置过期时间，避免死锁
        if conn.set(lockname, identifier, ex=lock_timeout, nx=True):
            return identifier

        time.sleep(0.001)

    return False


def release_lock(conn, lock_name, identifier):
    """
    释放锁
    
    :param conn: Redis 连接
    :param lockname: 锁的名称
    :param identifier: 锁的标识
    :return:
    """
    unlock_script = """
    if redis.call("get",KEYS[1]) == ARGV[1] then
        return redis.call("del",KEYS[1])
    else
        return 0
    end
    """
    lockname = f'lock:{lock_name}'
    unlock = conn.register_script(unlock_script)
    result = unlock(keys=[lockname], args=[identifier])
    if result:
        return True
    else:
        return False