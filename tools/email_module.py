import json  
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_naver(recipient="gudwls5789@gmail.com", 
                     subject="Email Sent Using Naver Mail",
                     path1="../results/parsed_data.json",
                     path2="../results/rag_results.json",
                     sender="gudwls5789@naver.com",
                     password="00",
                     smtp_server="smtp.naver.com",
                     smtp_port=465):
    """
    Reads two JSON files and combines the data based on index.
    The responses in parsed_data.json include 'label' and 'reason', 
    while rag_results.json provides 'clause' (article) and 'related_laws' information.
    
    In the output, 'clause' is displayed before 'reason'.
    The email body is composed in both HTML and plain text formats.
    
    Parameters:
        recipient (str): Recipient's email address.
        subject (str): Email subject.
        path1 (str): File path for the parsed_data JSON.
        path2 (str): File path for the rag_results JSON.
        sender (str): Sender's Naver email address.
        password (str): Sender's email password or app password.
        smtp_server (str): SMTP server address.
        smtp_port (int): SMTP port (default is 465 for SSL).
    """
    try:
        with open(path1, "r", encoding="utf-8") as f1:
            parsed_data = json.load(f1)
        with open(path2, "r", encoding="utf-8") as f2:
            rag_data = json.load(f2)
    except Exception as e:
        print("Error reading JSON files:", e)
        return

    # Create a dictionary for each index from parsed_data with label and reason
    parsed_dict = {}
    for entry in parsed_data:
        idx = entry.get("idx")
        responses = entry.get("responses", [])
        if responses:
            label = responses[0].get("label", "No label")
            reason = responses[0].get("reason", "No reason")
            parsed_dict[idx] = {"label": label, "reason": reason}
        else:
            parsed_dict[idx] = {"label": "No label", "reason": "No response available."}

    # Extract 'clause' (article) and 'related_laws' for each index from rag_results
    rag_results_list = rag_data.get("results", [])
    rag_clause_dict = {}
    rag_laws_dict = {}
    for entry in rag_results_list:
        idx = entry.get("index")
        clause = entry.get("clause")
        if clause:
            rag_clause_dict[idx] = clause
        related_laws = entry.get("related_laws", [])
        related_laws_text = "<br>".join(related_laws) if related_laws else "No related laws."
        rag_laws_dict[idx] = related_laws_text

    # Combine all indices to create the plain text email body
    combined_blocks = []
    all_indices = sorted(set(parsed_dict.keys()).union(set(rag_clause_dict.keys()), set(rag_laws_dict.keys())))
    for idx in all_indices:
        block = f"Index: {idx}\n"
        if idx in parsed_dict:
            block += "Responses:\n"
            block += f"Label: {parsed_dict[idx]['label']}\n"
            if idx in rag_clause_dict:
                block += f"Clause: {rag_clause_dict[idx]}\n"
            block += f"Reason: {parsed_dict[idx]['reason']}\n"
        if idx in rag_laws_dict:
            block += "Related Laws:\n" + rag_laws_dict[idx].replace("<br>", "\n") + "\n"
        combined_blocks.append(block)
    plain_body = "\n\n".join(combined_blocks)

    # Create HTML formatted email body with CSS styling
    html_body = """
    <html>
      <head>
         <meta charset="UTF-8">
         <style>
            .block {
                border: 1px solid #ddd;
                padding: 10px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .title {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
            .section-title {
                font-size: 16px;
                font-weight: bold;
                color: #555;
                margin-top: 10px;
            }
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
            }
         </style>
      </head>
      <body>
         <h2>Email Summary</h2>
    """
    for idx in all_indices:
        html_body += f'<div class="block">'
        html_body += f'<div class="title">조항 {idx}</div>'
        if idx in parsed_dict:
            if idx in rag_clause_dict:
                html_body += f'{rag_clause_dict[idx]}<br>'
            html_body += f'<div class="section-title">분석기의 판단: {parsed_dict[idx]["label"]}</div>'
            html_body += f'{parsed_dict[idx]["reason"]}</p>'
        if idx in rag_laws_dict:
            html_body += f'<div class="section-title">관련 법률 조항:</div>'
            html_body += f'<p>{rag_laws_dict[idx]}</p>'
        html_body += '</div>'
    html_body += """
      </body>
    </html>
    """

    # Create email MIME message (attach both plain text and HTML versions)
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    part1 = MIMEText(plain_body, "plain", "utf-8")
    part2 = MIMEText(html_body, "html", "utf-8")
    msg.attach(part1)
    msg.attach(part2)

    try:
        # Connect to the SMTP server using SSL
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
        server.quit()
        print("Email has been sent successfully.")
    except Exception as e:
        print("An error occurred while sending the email:", e)


if __name__ == "__main__":
    send_email_naver()
