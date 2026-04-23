USER_MAP = {
    "sunit": "19",
    "meenakshi": "30",
}

def get_writer_id(user_input: str):
    key = user_input.strip().lower()
    return USER_MAP.get(key, user_input.strip())
