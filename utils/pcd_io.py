import lzf
import time
import struct
import numpy as np
from io import StringIO

numpy_pcd_type_mappings = [
    (np.dtype('float32'), ('F', 4)),
    (np.dtype('float64'), ('F', 8)),
    (np.dtype('uint8'), ('U', 1)),
    (np.dtype('uint16'), ('U', 2)),
    (np.dtype('uint32'), ('U', 4)),
    (np.dtype('uint64'), ('U', 8)),
    (np.dtype('int16'), ('I', 2)),
    (np.dtype('int32'), ('I', 4)),
    (np.dtype('int64'), ('I', 8))
]


pcd_to_numpy_types_dict = {item[1]: item[0] for item in numpy_pcd_type_mappings}
numpy_to_pcd_types_dict = {item[0]: item[1] for item in numpy_pcd_type_mappings}


def seprate_rgb(rgb_val):
    rgb_val = rgb_val % 16777216
    b = rgb_val // 65536
    rgb_val = rgb_val % 65536
    g = rgb_val // 256
    r = rgb_val % 256
    return b, g, r


def load_pcd_from_bytes(pcd_bytes):
    pcd_byte_pos = 0

    fields = list()
    sizes = list()
    data_types = list()
    width = 0
    height = 0
    num_points = 0
    data_format = "ascii"

    while True:
        line = bytearray()
        while pcd_byte_pos < len(pcd_bytes):
            cur_byte = pcd_bytes[pcd_byte_pos]
            pcd_byte_pos += 1
            line.append(cur_byte)
            if cur_byte == 10:
                break
        if line.startswith(b"FIELDS"):
            line = line.decode(encoding="ascii")
            fields = [item.strip() for item in line.split(" ")[1:]]
        elif line.startswith(b"SIZE"):
            line = line.decode(encoding="ascii")
            sizes = [int(item) for item in line.split(" ")[1:]]
            for size in sizes:
                if size not in [1, 2, 4, 8]:
                    raise ValueError("Unknown Data Length: %d" % size)
        elif line.startswith(b"TYPE"):
            line = line.decode(encoding="ascii")
            data_types = [item.strip() for item in line.split(' ')[1:]]
            for data_type in data_types:
                if data_type not in ["F", "U"]:
                    raise TypeError("Unknown Data Type: %s" % data_type)
        elif line.startswith(b"WIDTH"):
            width = int(line.decode(encoding="ascii").split(' ')[1])
        elif line.startswith(b"HEIGHT"):
            height = int(line.decode(encoding="ascii").split(' ')[1])
        elif line.startswith(b"POINTS"):
            num_points = int(line.decode(encoding="ascii").split(' ')[1])
        elif line.startswith(b"DATA"):
            line = line.decode(encoding="ascii")
            data_format = line.split(' ')[1].strip()
            if data_format not in ["binary_compressed", "binary", "ascii"]:
                raise ValueError("unknown data_format: %s" % data_format)
            break
        else:
            continue
    
    datas = dict()

    if "binary" == data_format:
        buf = pcd_bytes[pcd_byte_pos:]
        buf = buf[:sum(sizes) * num_points]
        
        np_typenames = [pcd_to_numpy_types_dict[(data_type, size)] for size, data_type in zip(sizes, data_types)]
        dtype = np.dtype(list(zip(fields, np_typenames)))

        fields_np_data = np.frombuffer(buf, dtype=dtype)

        datas = dict()
        for field in fields:
            if field == "rgb":
                r, g, b = seprate_rgb(fields_np_data[field])
                datas["r"] = r
                datas["g"] = g
                datas["b"] = b
            else:
                datas[field] = fields_np_data[field]
    elif data_format == "binary_compressed":
        fmt = 'II'
        compressed_size, uncompressed_size =\
        struct.unpack(fmt, pcd_bytes[pcd_byte_pos:pcd_byte_pos+struct.calcsize(fmt)])
        pcd_byte_pos += struct.calcsize(fmt)
        compressed_data = pcd_bytes[pcd_byte_pos:pcd_byte_pos+compressed_size]
        
        buf = lzf.decompress(compressed_data, uncompressed_size)

        datas = dict()
        start_pos = 0
        for field, size, data_type in zip(fields, sizes, data_types):
            field_buf_length = size * num_points
            field_buf = buf[start_pos:start_pos + field_buf_length]
            start_pos += field_buf_length

            field_data = np.frombuffer(field_buf, dtype=pcd_to_numpy_types_dict[(data_type, size)])

            if field == "rgb":
                r, g, b = seprate_rgb(field_data)
                datas["r"] = r
                datas["g"] = g
                datas["b"] = b
            else:
                datas[field] = field_data
    elif data_format == "ascii":
        buf = pcd_bytes[pcd_byte_pos:]
        txt = buf.decode("ascii").strip()
        np_typenames = [pcd_to_numpy_types_dict[(data_type, size)] for size, data_type in zip(sizes, data_types)]
        
        dtype = np.dtype(list(zip(fields, np_typenames)))
        fields_np_data = np.loadtxt(StringIO(txt), dtype=dtype, delimiter=" ")

        datas = dict()
        for field in fields:
            if field == "rgb":
                r, g, b = seprate_rgb(fields_np_data[field])
                datas["r"] = r
                datas["g"] = g
                datas["b"] = b
            else:
                datas[field] = fields_np_data[field]
    return datas, fields, sizes, data_types


def load_pcd(pcd_path):
    file = open(pcd_path, "rb")
    pcd_bytes = file.read()
    datas, fields, sizes, data_types = load_pcd_from_bytes(pcd_bytes)
    return datas, fields, sizes, data_types


def save_pcd(pcd_path, pcd_data, fields, sizes, data_types, pcd_type):
    num_points = pcd_data[list(pcd_data.keys())[0]].shape[0]
    # pcd_type: ascii binary binary_compressed
    file = open(pcd_path, "wb")
    file.write(b"# .PCD v0.7 - Point Cloud Data file format\n")
    file.write(b"VERSION 0.7\n")
    file.write(("FIELDS " + " ".join(fields) + "\n").encode(encoding="ascii"))
    file.write(("SIZE " + " ".join([str(item) for item in sizes]) + "\n").encode(encoding="ascii"))
    file.write(("TYPE " + " ".join(data_types) + "\n").encode(encoding="ascii"))
    file.write(("COUNT " + " ".join(["1" for _ in range(len(fields))]) + "\n").encode(encoding="ascii"))
    file.write(("WIDTH %d\n" % num_points).encode(encoding="ascii"))
    file.write(b"HEIGHT 1\n")
    file.write(b"VIEWPOINT 0 0 0 1 0 0 0\n")
    file.write(("POINTS %d\n" % num_points).encode(encoding="ascii"))
    file.write(("DATA %s\n" % pcd_type).encode(encoding="ascii"))
    if "ascii" == pcd_type:
        for i in range(num_points):
            items = list()
            for field, data_type in zip(fields, data_types):
                if field != "rgb":
                    if data_type == "F":
                        items.append("%f" % float(pcd_data[field][i]))
                    elif data_type == "U":
                        items.append("%d" % int(pcd_data[field][i]))
                else:
                    r =  int(pcd_data["r"][i]) % 256
                    g =  int(pcd_data["g"][i]) % 256
                    b =  int(pcd_data["b"][i]) % 256
                    rgb = r  + g * 256 + b * 65536
                    items.append("%d" % rgb)
            line = " ".join(items) + "\n"
            file.write(line.encode(encoding="ascii"))
    elif "binary" == pcd_type:
        buf = bytearray()
        for i in range(num_points):
            for field, data_type, size in zip(fields, data_types, sizes):
                if field != "rgb":
                    if size == 8 and data_type == "F":
                        buf += struct.pack("<d", float(pcd_data[field][i]))
                    elif size == 4 and data_type == "F":
                        buf += struct.pack("<f", float(pcd_data[field][i]))
                    elif size==4 and data_type == "U":
                        buf += struct.pack("<I", int(pcd_data[field][i]))
                    elif size==2 and data_type == "U":
                        buf += struct.pack("<H", int(pcd_data[field][i]))
                    elif size==1 and data_type == "U":
                        buf += struct.pack("<B", int(pcd_data[field][i]))
                    else:
                        raise TypeError("unknown data_type %s with data_size %d" % (data_type, size))
                else:
                    r =  pcd_data["r"][i]
                    g =  pcd_data["g"][i]
                    b =  pcd_data["b"][i]
                    rgb = r  + g * 256 + b * 65536
                    buf += struct.pack("<I", int(rgb))
        file.write(buf)
    elif "binary_compressed" == pcd_type:
        buf = bytearray()
        for field, data_type, size in zip(fields, data_types, sizes):
            for i in range(num_points):
                if field != "rgb":
                    if size == 8 and  data_type == "F":
                        buf += struct.pack("<d", float(pcd_data[field][i]))
                    elif size == 4 and data_type == "F":
                        buf += struct.pack("<f", float(pcd_data[field][i]))
                    elif size==4 and data_type == "U":
                        buf += struct.pack("<I", int(pcd_data[field][i]))
                    elif size==2 and data_type == "U":
                        buf += struct.pack("<H", int(pcd_data[field][i]))
                    elif size==1 and data_type == "U":
                        buf += struct.pack("<B", int(pcd_data[field][i]))
                    else:
                        raise TypeError("unknown data_type %s with data_size %d" % (data_type, size))
                else:
                    r =  pcd_data["r"][i]
                    g =  pcd_data["g"][i]
                    b =  pcd_data["b"][i]
                    rgb = r  + g * 256 + b * 65536
                    buf += struct.pack("<I", int(rgb))
        buf = bytes(buf)
        compressed_buf = lzf.compress(buf)
        data_buf = struct.pack("II", len(compressed_buf), len(buf)) + compressed_buf
        file.write(data_buf)
    else:
        raise TypeError("unknown pcd_type: %s" % pcd_type)
    file.close()
