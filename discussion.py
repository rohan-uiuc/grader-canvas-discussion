import requests
import json

# Replace these variables with your own information
access_token = 'YOUR_ACCESS_TOKEN'
course_id = '36263'
discussion_topic_id = '421517'
base_url = 'https://canvas.illinois.edu'

headers = {
    'Authorization': f'Bearer {access_token}'
}

# Create a content export
export_url = f'{base_url}/api/v1/courses/{course_id}/content_exports'
export_params = {
    'export_type': 'common_cartridge',
    'skip_notifications': True,
    'select': {
        'discussion_topics': [discussion_topic_id]
    }
}

export_response = requests.post(export_url, headers=headers, params=export_params)

if export_response.ok:
    export_data = export_response.json()
    export_id = export_data['id']

    # Check the progress of the content export
    progress_url = f'{base_url}/api/v1/progress/{export_id}'
    progress_response = requests.get(progress_url, headers=headers)

    if progress_response.ok:
        progress_data = progress_response.json()
        while progress_data['workflow_state'] not in ['completed', 'failed']:
            progress_response = requests.get(progress_url, headers=headers)
            progress_data = progress_response.json()

        if progress_data['workflow_state'] == 'completed':
            # Download the exported content
            download_url = progress_data['url']
            download_response = requests.get(download_url)

            if download_response.ok:
                # Save the exported content to a file
                with open('discussion_topic_export.imscc', 'wb') as f:
                    f.write(download_response.content)
else:
    print(f'Error: {export_response.text}')
