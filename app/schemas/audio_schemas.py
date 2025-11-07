from pydantic import BaseModel

class StemInfo(BaseModel):
    file_id: str
    filename: str
    download_url: str

class SeparationResult(BaseModel):
    result: dict[str, StemInfo]