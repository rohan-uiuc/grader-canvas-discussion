import requests
import json
import os
from typing import List

class DiscussionEntry:
    def __init__(self, id: int, parent_id: int, name: str, message: str, replies: List):
        self.id = id
        self.parent_id = parent_id
        self.name = name
        self.message = message
        self.replies = replies

    def to_json(self):
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'name': self.name,
            'message': self.message,
            'replies': [reply.to_json() for reply in self.replies]
        }

def extract_entries(entries, participants):
    result = []
    for entry in entries:
        if 'message' in entry and 'deleted' not in entry:
            id = entry['id']
            parent_id = entry['parent_id']
            user_id = entry['user_id']
            name = next((p['display_name'] for p in participants if p['id'] == user_id), None)
            message = entry['message']
        replies = []
        if 'replies' in entry:
            replies = extract_entries(entry['replies'], participants)
        result.append(DiscussionEntry(id, parent_id, name, message, replies))
    return result

def save_messages(entries):

    for entry in entries:
        # Save the message as an HTML file
        filename = f'docs/{entry.name}.html'

        # Open file in write/append mode
        with open(filename, 'a+') as f:
            if  entry.parent_id == None:
                f.write(f'<p><b>Student Post: {entry.name}</b></p>')
                f.write(entry.message)
                f.write('<hr>')
            else:
                f.write(f'<p><b>Reply to: {entry.parent_id}</b></p>')
                f.write(entry.message)
                f.write('<hr>')


        # Save the messages of the replies
    for entry in entries:

        save_messages(entry.replies)

# Replace these variables with your own information
access_token = ''
course_id = '36263'
discussion_topic_id = '421517'
base_url = 'https://canvas.illinois.edu'

headers = {
    'Authorization': f'Bearer {access_token}'
}

# Retrieve the full discussion topic data
discussion_url = f'{base_url}/api/v1/courses/{course_id}/discussion_topics/{discussion_topic_id}/view'
discussion_response = requests.get(discussion_url, headers=headers)

if discussion_response.ok:
    discussion_data = discussion_response.json()
    with open('discussion_data.json', 'w') as f:
        json.dump(discussion_data, f)

    # Extract the desired fields from the replies and responses
    entries = extract_entries(discussion_data['view'], discussion_data['participants'])

    # Save the extracted data to a file
    with open('discussion_entries.json', 'w') as f:
        json.dump([entry.to_json() for entry in entries], f)

    # Create the /docs directory if it does not exist
    os.makedirs('docs', exist_ok=True)

    # Save the messages as HTML files under the /docs directory
    save_messages(entries)

    # Extract the rubric and save it to a file
    if 'rubric' in discussion_data:
        rubric = discussion_data['rubric']
        with open('rubric.json', 'w') as f:
            json.dump(rubric, f)
else:
    print(f'Error: {discussion_response.text}')

rubric_url = f'{base_url}/api/v1/courses/{course_id}/discussion_topics/{discussion_topic_id}'
rubric_response = requests.get(rubric_url, headers=headers)

if rubric_response.ok:
    rubric_data = rubric_response.json()
    # print(rubric_data)
    if 'rubric' in rubric_data['assignment']:
        rubric = rubric_data['assignment']['rubric']
        with open('rubric_data.json', 'w') as f:
            json.dump(rubric, f)