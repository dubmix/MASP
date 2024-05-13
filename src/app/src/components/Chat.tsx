import React, { useState } from 'react';
import '../styles/Chat.css';

const Chat: React.FC = () => {
    const [messageInput, setMessageInput] = useState<string>('');
    const [showSavedMessage, setShowSavedMessage] = useState<boolean>(false);

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!messageInput.trim()) return;
        displayUserChatBubble(messageInput);
        setMessageInput('');

        try {
            const response = await fetch('http://127.0.0.1:5000/api/process-request', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ request: messageInput })
            });

            if (response.ok) {
                const responseData = await response.json();
                const backendResponse = responseData.answer;
                displayServerChatBubble(backendResponse)
            } else {
                throw new Error('Connection with backend server failed');
            }
        } catch (error) {
            console.error('Error sending message: ', error);
            displayServerChatBubble('Connection with backend server failed');
        }
    };

    const displayServerChatBubble = (message: string) => {
        const chatMessagesElement = document.getElementById('chat-container')!;
        const chatBubbleElement = createChatBubble(message, 'server-bubble');
        chatMessagesElement.insertBefore(chatBubbleElement, chatMessagesElement.firstChild);
    }

    const displayUserChatBubble = (message: string) => {
        const chatMessagesElement = document.getElementById('chat-container')!;
        const chatBubbleElement = createChatBubble(message, 'user-bubble');
        chatMessagesElement.insertBefore(chatBubbleElement, chatMessagesElement.firstChild);
    }

    const handleDoubleClick = () => {
        setShowSavedMessage(true);
        setTimeout(() => {setShowSavedMessage(false)}, 2000);
    };

    const createChatBubble = (message: string, bubbleType: string) => {
        const chatBubbleElement = document.createElement('div');
        chatBubbleElement.className = `chat-bubble ${bubbleType}`;
        chatBubbleElement.innerHTML = message;

        if (bubbleType === 'server-bubble') {
            chatBubbleElement.ondblclick = () => handleDoubleClick();
        }
        return chatBubbleElement;
    }

    return (
        <>
        <div id="chat-container">
        </div>

        {showSavedMessage && <div className="saved-message">Message saved!</div>}

        <form id="messageForm" onSubmit={handleSubmit}>
            <div className="input-container">
                <input type="text" 
                    id="messageInput" 
                    placeholder="Ask me anything..."
                    value={messageInput}
                    autoComplete='off'
                    onChange={(e) => setMessageInput(e.target.value)} />
                <button type="submit">Send</button>
            </div>
        </form>
        </>
    )
};

export default Chat;