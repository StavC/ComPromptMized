import email
import os
import socket
import threading
from queue import Queue
import Run_LLaVa

# Server configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 1025
saveMail_directory = "FlowSteering/ApplicationCode/LLaVaServer/EmailLLaVaMailDatabase"
MODEL_NAME = "FlowSteering/llava/llava_weights/"  # PATH to the LLaVA weights
message_queue = Queue()
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


def handle_messages():  # This function is used to handle the messages in the queue and process them accordingly based on the command received from the client

    while True:
        if not message_queue.empty():

            print('______________________________________________________________')

            data, client_socket, client_address, model, image_processor, tokenizer, device = message_queue.get()

            msg = email.message_from_bytes(data)

            Command, subject, sender, recipient = msg['Command'], msg["Subject"], msg["From"], msg["To"]

            if Command == "CHECK_EMAIL":
                print("Sending the Email to LLaVa model for classification")
                SendToLLaVa(data, client_socket, sender, recipient, subject, model, image_processor, tokenizer, device) # This command is used to request the LLaVa server to send the email to the LLaVa model for classification.

            print('______________________________________________________________')
            client_socket.close()


def SendToLLaVa(data, client_socket, sender, recipient, subject, model, image_processor, tokenizer, device): # This function is used to send the email to the LLaVa model for classification
    recipient_directory = f"{saveMail_directory}/{recipient}"
    os.makedirs(recipient_directory, exist_ok=True)

    msg = email.message_from_bytes(data)

    if msg.is_multipart():
        for part in msg.get_payload():
            if part.get_content_type() == "text/plain":
                body = part.get_payload()

    else:
        print(msg.get_payload())
    # print the subject
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        if part.get("Content-Disposition") is None:
            continue

        filename = part.get_filename()
        # split the filename by "\" and take the last part of it
        #filename = filename.split("\\")[-1]
        filename = filename.split("/")[-1]

        # Save the image file
        filepath = str(f"{recipient_directory}/{filename}")
        with open(filepath, "wb") as f:
            f.write(part.get_payload(decode=True))

    print(f"From: {sender}")
    print(f"To: {recipient}")
    print(f"Subject: {subject}")
    print(f"Attachment filename: {filename}")
    print(f' Text body: {body}')

    Query = body
    AdditionalQueryNum = msg['AdditionalQueryNum']
    AdditionalQueryNum = int(AdditionalQueryNum)
    query_list = []
    for i in range(AdditionalQueryNum):
        AdditionalQuery = msg[f'AdditionalQuery{str(i)}']
        AdditionalQuery = AdditionalQuery.replace('-@-', '\n')  # replace the -@- with a new line character, as we had some issues with the new line character in the client
        query_list.append(AdditionalQuery)

    tokenizer, image_processor, vision_tower, unorm, norm, embeds, projector, prompt, input_ids = Run_LLaVa.load_param(
        MODEL_NAME, model, tokenizer, Query)

    reply = Run_LLaVa.Run_LLaVa(filepath, prompt, Query, query_list, model, tokenizer, unorm, image_processor) # Run the LLaVa model on the email and the additional queries and get the response from the model

    FinalReply = ''
    for i in range(len(reply)):
        FinalReply += f'Response {i}: {reply[i]}'

    FinalReply = FinalReply.encode('ascii', 'ignore').decode('ascii') # encode the reply to ascii and ignore any characters that can't be encoded

    client_socket.sendall(FinalReply.encode('utf-8'))
    client_socket.close()
    print(f'sent a reply to the client {recipient}')
    print('______________________________________________________________')


def start_server(): # This function is used to start the server and listen for incoming connections
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(1000)
    model, image_processor, tokenizer, device = Run_LLaVa.Turn_On_LLaVa() # Turn on the LLaVa model and get the model, image processor, tokenizer and the device

    print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")

    threading.Thread(target=handle_messages, daemon=True).start()

    while True:
        client_socket, client_address = server_socket.accept()
        data = receive_complete_data(client_socket)

        if data:
            print(f"Received message from {client_address} put in queue")
            # Put the data in the queue
            message_queue.put((data, client_socket, client_address, model, image_processor, tokenizer, device))


if __name__ == '__main__':
    os.makedirs(saveMail_directory, exist_ok=True)
    start_server()
