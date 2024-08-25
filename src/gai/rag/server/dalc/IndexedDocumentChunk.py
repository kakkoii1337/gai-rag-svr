from sqlalchemy import Column, ForeignKey, Text, VARCHAR, DateTime, Boolean, BLOB, JSON, INTEGER, Date
from gai.rag.server.dalc.Base import Base
from sqlalchemy.orm import relationship

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