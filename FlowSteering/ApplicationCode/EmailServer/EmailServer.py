import email
import os
import socket
import threading
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from queue import Queue

import pandas as pd

# Server configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 1234
saveMail_directory = "FlowSteering/ApplicationCode/EmailServer/EmailServerMailDatabase"  # Change this to the directory where you want to save the emails inbox for each user
message_queue = Queue()
default_image = 'FlowSteering/assets/PerturbatedImages/DjiPerturbClassForward.png'
# Server configuration

def receive_complete_data(
        client_socket):  # This function is used to receive the complete data from the client, adjust the parameters as needed based on your network conditions
    received_data = b""
    count = 0
    client_socket.settimeout(3.0)
    try:
        while True:
            chunk = client_socket.recv(2 ** 16)  # Adjust the buffer size as needed
            if not chunk:
                count += 1
            else:
                count = 0
                received_data += chunk
            if count >= 50:
                break

    except socket.timeout as e:
        print('timeout')
        print(e)

        pass

    return received_data


def handle_messages():  # This function is used to handle the messages in the queue and process them accordingly based on the command received from the client (e.g., SEND_EMAIL, CHECK_INBOX)
    while True:
        if not message_queue.empty():

            print('______________________________________________________________')

            data, client_socket, client_address = message_queue.get()

            msg = email.message_from_bytes(data)

            Command, subject, sender, recipient = msg['Command'], msg["Subject"], msg["From"], msg["To"]

            if Command == "CHECK_INBOX":
                print("Checking Inbox")
                Check_Inbox(client_socket,
                            sender)  # This function is used to check the inbox of the user and send the email to the client
            elif Command == "SEND_EMAIL":  # This is the command to send the email to the recipient
                print("Sending Email")
                Save_Email_To_Recipient(client_socket, data, msg, Command, subject, sender,
                                        recipient)  # This function is used to save the email to the recipient's inbox
            print('______________________________________________________________')
            client_socket.close()


def Save_Email_To_Recipient(client_socket, data, msg, requests, subject, sender, recipient): # This function is used to save the email to the recipient's inbox
    recipient_directory = f"{saveMail_directory}/{recipient}" # This is the directory where the emails will be saved
    os.makedirs(recipient_directory, exist_ok=True) # Create the directory if it doesn't exist

    msg = email.message_from_bytes(data)

    if msg.is_multipart():
        for part in msg.get_payload():
            if part.get_content_type() == "text/plain":
                body = part.get_payload()

    else:
        print(msg.get_payload())

    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        if part.get("Content-Disposition") is None:
            continue

        # Get the filename
        filename = part.get_filename()
        # split the filename by "\" and take the last part of it
        #filename = filename.split("\\")[-1]
        filename = filename.split("/")[-1]

        # Save the image file
        with open(os.path.join(recipient_directory, filename), "wb") as f:
            f.write(part.get_payload(decode=True))

    print(f"From: {sender}")
    print(f"To: {recipient}")
    print(f"Subject: {subject}")
    print(f"Attachment filename: {filename}")
    print(f' Text body: {body}')


    filepath = str(f"{recipient_directory}/{filename}")

    email_data = [[sender, recipient, subject, body, filepath]]

    MyColumns = ['Sender', 'Recipient', 'Subject', 'Body', 'FilePath']
    if not os.path.isfile(f"{recipient_directory}/{recipient}_received_emails.csv") or (
            os.stat(f"{recipient_directory}/{recipient}_received_emails.csv").st_size == 0): # If the file doesn't exist, then create the file and save the email to the file
        df = pd.DataFrame(email_data, columns=MyColumns)
        df.to_csv(f"{recipient_directory}/{recipient}_received_emails.csv", mode='w', header=True, index=False) # Save the email to the recipient's inbox
        df.to_csv(f"{recipient_directory}/{recipient}_received_emailsHistory.csv", mode='w', header=True, index=False) # Save the email to the recipient's inbox history

    else: # If the file already exists, then append the email to the file

        df = pd.read_csv(f"{recipient_directory}/{recipient}_received_emails.csv") # Read the csv file of the recipient
        new_row_df = pd.DataFrame(email_data, columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(f"{recipient_directory}/{recipient}_received_emails.csv", mode='w', header=True, index=False)
        df = pd.read_csv(f"{recipient_directory}/{recipient}_received_emailsHistory.csv")
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(f"{recipient_directory}/{recipient}_received_emailsHistory.csv", mode='w', header=True, index=False)

    # write back to the sender that the email was sent
    client_socket.sendall("Email Sent".encode('utf-8'))


def Check_Inbox(client_socket, sender): # This function is used to check the inbox of the user and send the email to the client

    print(f' A request ot check the inbox email from: {sender}')

    sender_directory = f"{saveMail_directory}/{sender}"
    os.makedirs(sender_directory, exist_ok=True)

    if (not os.path.isfile(f"{sender_directory}/{sender}_received_emails.csv")) or (
            os.stat(f"{sender_directory}/{sender}_received_emails.csv").st_size == 0):
        client_socket.sendall("No Emails".encode('utf-8'))
        return
    df = pd.read_csv(f"{sender_directory}/{sender}_received_emails.csv")
    rows = df.shape[0]
    print(f'found {rows} emails in the inbox of {sender}')
    if rows == 0: # If there are no emails in the inbox, then send "No Emails" to the client
        client_socket.sendall("No Emails".encode('utf-8'))
        return
    else: # If there are emails in the inbox, then send the email to the client
        # take the last row of the csv file
        header_columns = df.columns
        last_row = df.tail(1)
        msg = MIMEMultipart()
        msg["Command"] = "SEND_EMAIL"
        msg["From"] = last_row['Sender'].values[0]
        msg["To"] = last_row['Recipient'].values[0]
        msg["Subject"] = last_row['Subject'].values[0]
        msg.attach(MIMEText(last_row['Body'].values[0], "plain"))

        filename = last_row['FilePath'].values[0]
        with open(filename, "rb") as f:
            try:  #We faced some network errors resulting in images being sent partially black. To address this issue, we implemented a try-except block to handle such occurrences. Now, if an image fails to send correctly, a default image is sent for that experiment.
                img = MIMEImage(f.read())
                img.add_header("Content-Disposition", "attachment", filename=filename)
                msg.attach(img)
            except:
                print('network error, sending default image instead of the original image')
                with open(default_image,"rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header("Content-Disposition", "attachment", filename=filename)
                    msg.attach(img)

        message = msg.as_bytes()
        # send the message to the client
        df.drop(df.tail(1).index, inplace=True)

        df.to_csv(f"{sender_directory}/{sender}_received_emails.csv", mode='w', header=True, index=False)
        client_socket.sendall(message)
        return


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(1000)

    print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")

    threading.Thread(target=handle_messages, daemon=True).start()

    while True:
        client_socket, client_address = server_socket.accept()
        print(len(message_queue.queue))

        # Receive complete data from the client
        data = receive_complete_data(client_socket)

        if data:
            print(f"Received message from {client_address} put in queue")
            message_queue.put((data, client_socket, client_address))


if __name__ == '__main__':
    os.makedirs(saveMail_directory, exist_ok=True)
    start_server()
