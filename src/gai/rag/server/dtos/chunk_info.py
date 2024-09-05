from pydantic import BaseModel, ConfigDict, Field

# The ChunkInfo class is a subset of the Chunk model.
# 
class ChunkInfoPydantic(BaseModel):
    Id: str = Field(...)  # The ellipsis here indicates a required field with no default value
    ChunkHash: str = Field(...)  # The ellipsis here indicates a required field with no default value
    IsDuplicate: bool
    IsIndexed: bool
    model_config = ConfigDict(from_attributes=True)
