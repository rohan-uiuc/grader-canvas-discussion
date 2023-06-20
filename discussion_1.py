import requests
import json

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

    # Extract the replies and responses
    discussions = []
    replies = []
    for entry in discussion_data['view']:
        discussions.extend(entry)
        if 'replies' in entry:
            replies.extend(entry['replies'])

    with open('discussions.json', 'w') as f:
        json.dump(discussions, f)

    # Save the replies and responses to a file
    with open('discussion_replies.json', 'w') as f:
        json.dump(replies, f)
else:
    print(f'Error: {discussion_response.text}')
