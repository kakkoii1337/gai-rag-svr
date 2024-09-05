from sqlalchemy import Column, ForeignKey, Text, VARCHAR, DateTime, Boolean, BLOB, JSON, INTEGER, Date
from gai.rag.server.dalc.Base import Base
from sqlalchemy.orm import relationship
from gai.rag.dtos.indexed_doc_chunk import IndexedDocChunkPydantic

class IndexedDocumentChunk(Base):
    __tablename__ = 'IndexedDocumentChunks'

    Id = Column(VARCHAR(36), primary_key=True)
    ChunkHash = Column(VARCHAR(64))   # SHA256 hash of the chunk
    ChunkGroupId = Column(VARCHAR(36), ForeignKey('IndexedDocumentChunkGroups.Id'))
    ByteSize = Column(INTEGER)
    IsDuplicate = Column(Boolean)
    IsIndexed = Column(Boolean)
    Content = Column(Text)

    # Relationship to IndexedDocumentChunkGroup
    ChunkGroup = relationship("IndexedDocumentChunkGroup", back_populates="Chunks")

    def to_pydantic(self):
        try:
            return IndexedDocChunkPydantic(**{
                'Id': self.Id,
                'ChunkHash': self.ChunkHash,
                'ChunkGroupId': self.ChunkGroupId,
                'ByteSize': self.ByteSize,
                'IsDuplicate': self.IsDuplicate,
                'IsIndexed': self.IsIndexed,
                'Content': self.Content
            })
        except Exception as e:
            raise RuntimeError("Failed to convert from DALC to Pydantic model.") from e