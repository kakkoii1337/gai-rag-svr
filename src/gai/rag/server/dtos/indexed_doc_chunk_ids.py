from pydantic import BaseModel


class IndexedDocChunkIdsPydantic(BaseModel):
    DocumentId: str
    ChunkgroupId: str
    ChunkIds: list[str]
