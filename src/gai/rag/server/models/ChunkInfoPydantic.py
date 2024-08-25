from pydantic import BaseModel, ConfigDict, Field

# The ChunkInfo class is a subset of the Chunk model.
# 
class ChunkInfoPydantic(BaseModel):
    Id: str = Field(...)  # The ellipsis here indicates a required field with no default value
    ChunkHash: str = Field(...)  # The ellipsis here indicates a required field with no default value
    IsDuplicate: bool
    IsIndexed: bool
    model_config = ConfigDict(from_attributes=True)

    @staticmethod
    def from_dalc(orm):
        return ChunkInfoPydantic(
            Id=orm.Id,
            ChunkHash=orm.ChunkHash,
            IsDuplicate=orm.IsDuplicate,
            IsIndexed=orm.IsIndexed
        )