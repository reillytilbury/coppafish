import smtplib


def send_email(subject: str, body: str, sender: str, recipient: str, password: str) -> None:
    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.starttls()
    s.login(sender, password)
    message = "Subject: {}\n\n{}".format(subject, body)
    s.sendmail(sender, recipient, message)
    s.quit()
