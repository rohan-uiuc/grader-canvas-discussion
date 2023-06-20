from pydantic import BaseModel
from typing import List, Optional

class ForumUser(BaseModel):
    id: int
    anonymous_id: str
    display_name: str
    avatar_image_url: str
    html_url: str
    pronouns: Optional[str]

class ForumPost(BaseModel):
    id: int
    user_id: int
    parent_id: Optional[int]
    created_at: str
    updated_at: str
    rating_count: Optional[int]
    rating_sum: Optional[int]
    user_name: str
    message: str
    user: ForumUser
    read_state: str
    forced_read_state: bool

def get_data_from_json(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
        data = [ForumPost(**item) for item in json_data]
        return data