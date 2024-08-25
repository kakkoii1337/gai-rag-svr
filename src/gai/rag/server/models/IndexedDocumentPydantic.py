from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime,date
from gai.rag.server.models.IndexedDocumentChunkGroupPydantic import IndexedDocumentChunkGroupPydantic
from gai.rag.server.dalc.IndexedDocument import IndexedDocument

class IndexedDocumentPydantic(BaseModel):
    Id: str = Field(...)
    CollectionName: str
    ByteSize: int
    FileName: Optional[str] = None
    FileType: Optional[str] = None
    File: Optional[bytes] = None
    Source: Optional[str] = None
    Abstract: Optional[str] = None
    Authors: Optional[str] = None
    Title: Optional[str] = None
    Publisher: Optional[str] = None
    PublishedDate: Optional[date] = None
    Comments: Optional[str] = None
    Keywords: Optional[str] = None
    CreatedAt: datetime
    UpdatedAt: datetime
    IsActive: bool = True
    ChunkGroups: Optional[List[IndexedDocumentChunkGroupPydantic]] = None

    # class Config:
    def from_dalc(orm: IndexedDocument):
        return IndexedDocumentPydantic(
            Id=orm.Id,
            CollectionName=orm.CollectionName,
            ByteSize=orm.ByteSize,
            FileName=orm.FileName,
            FileType=orm.FileType,
            Source=orm.Source,
            Abstract=orm.Abstract,
            Authors=orm.Authors,
            Title=orm.Title,
            Publisher=orm.Publisher,
            PublishedDate=orm.PublishedDate,
            Comments=orm.Comments,
            Keywords=orm.Keywords,
            CreatedAt=orm.CreatedAt,
            UpdatedAt=orm.UpdatedAt,
            IsActive=orm.IsActive,
            File=orm.File,
            ChunkGroups=[IndexedDocumentChunkGroupPydantic.from_dalc(cg) for cg in orm.ChunkGroups]
        )