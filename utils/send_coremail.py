import smtplib
from email.mime.text import MIMEText

# MAIL_ADDR="brli@iflytek.com"
DST_ADDR="xuanliu11@iflytek.com"
SMTP_SERVER="mail.iflytek.com"
SMTP_PORT=25
SMTP_SSL_PORT=465
MAIL_ADDR="taoguo@iflytek.com"
PASSWD="Pwdg6T2PmnnEDAI5"
# PASSWD="wkuiuwd74rAMbp7S"

class MailHelper():
    def __init__(self, user=MAIL_ADDR, passwd=PASSWD) -> None:
        self.__mail_addr = user
        self.status = False
        try:
            self.handle =  smtplib.SMTP_SSL(SMTP_SERVER, SMTP_SSL_PORT)
            self.handle.login(self.__mail_addr, passwd)
            self.status = True
        except:
            print("Login to mail server failed.")

    # def __del__(self):
    #     if self.status:
    #         self.handle.quit()

    def send(self, subject, text, dst_user=[DST_ADDR] ):
        if not self.status:
            return False
        
        msg = MIMEText(text)
        msg['Subject'] = subject
        msg['From'] = self.__mail_addr
        msg['To'] = ",".join(dst_user)
        msg['Cc'] = self.__mail_addr

        dst_user.append(self.__mail_addr)
        self.handle.sendmail(self.__mail_addr, dst_user, msg.as_string())
        return True
    
mail_handle = MailHelper()

if __name__ == '__main__':
    mail = MailHelper()
    mail.send("[DR_NOTICE] sihao_27en6.20240322.reconstruct.done", "hello world", ["lerin123@163.com"])
