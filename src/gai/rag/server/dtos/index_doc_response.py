
from pydantic import BaseModel


class IndexDocResponse(BaseModel):
    DocumentId: str
    ChunkgroupId: str
    ChunkIds: list[str]