import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import json

class QQMailSender:
    def __init__(self, qq_email, auth_code):

        self.qq_email = qq_email
        self.auth_code = auth_code
        self.smtp_server = "smtp.qq.com"
        self.smtp_port_ssl = 465  # SSL端口
        self.smtp_port_tls = 587  # TLS端口
    
    def send_email_with_attachment(self, subject, attachment_path, use_ssl=True):

        to_emails = "" #替换为打印机的邮箱地址
        content = ""
        try:
            # 创建多部分邮件对象
            msg = MIMEMultipart()
            msg['From'] = self.qq_email
            msg['Subject'] = subject
            
            # 处理收件人
            if isinstance(to_emails, str):
                to_emails = [to_emails]
            msg['To'] = ', '.join(to_emails)
            
            # 添加邮件正文
            msg.attach(MIMEText(content, 'plain', 'utf-8'))
            
            # 添加附件
            if os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                filename = os.path.basename(attachment_path)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                msg.attach(part)
            else:
                print(f"警告: 附件文件不存在 - {attachment_path}")
            
            # 连接SMTP服务器并发送邮件
            if use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port_ssl)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port_tls)
                server.starttls()
            
            server.login(self.qq_email, self.auth_code)
            server.sendmail(self.qq_email, to_emails, msg.as_string())
            server.quit()
            
            print(f"带附件的邮件发送成功！收件人: {', '.join(to_emails)}")
            return True
            
        except Exception as e:
            print(f"带附件的邮件发送失败: {str(e)}")
            return False

# 使用示例
if __name__ == "__main__":
    # 配置你的QQ邮箱信息
    QQ_EMAIL = ""  # 替换为你的QQ邮箱
    AUTH_CODE = ""  # 替换为你的授权码
    
    # 创建邮件发送器
    mailer = QQMailSender(QQ_EMAIL, AUTH_CODE)

    functions = [
    {
        "name": "send_email_with_attachment",
        "description": "发送带附件的邮件",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "邮件主题"
                },
                "attachment_path": {
                    "type": "string",
                    "description": "附件路径"
                },
            },
            "required": ["subject", "attachment_path"]
        }
    }
]
    
    from openai import OpenAI
    client = OpenAI(
        api_key = "sk-10acd30d8f8c4fbd90b74ed43521b10f", 
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    user_input = "请发送教师报告，是路径在printer下的teacher_report.pdf文件"

    response = client.chat.completions.create(
        model='qwen3-32b',  # 或者 qwen-plus, qwen-max 等
        messages=[{'role': 'user', 'content': user_input}],
        functions=functions, 
        function_call="auto",
        extra_body = {"enable_thinking": False}
    )

    message = response.choices[0].message  

    if message.function_call:
        function_name = message.function_call.name
        arguments = json.loads(message.function_call.arguments)
        
        subject = arguments["subject"]
        attachment_path = arguments["attachment_path"]
        
        mailer.send_email_with_attachment(subject, attachment_path)