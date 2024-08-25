from sqlalchemy import Column, ForeignKey, Text, VARCHAR, DateTime, Boolean, BLOB, JSON, INTEGER, Date
from gai.rag.server.dalc.Base import Base
from sqlalchemy.orm import relationship

class IndexedDocumentChunkGroup(Base):
    __tablename__ = 'IndexedDocumentChunkGroups'

    # Primary key for the chunk group itself
    Id = Column(VARCHAR(36), primary_key=True)
    DocumentId = Column(VARCHAR(36), ForeignKey('IndexedDocuments.Id'), nullable=False)
    SplitAlgo = Column(VARCHAR(200))
    ChunkCount = Column(INTEGER, nullable=False)
    ChunkSize = Column(INTEGER, nullable=False)
    Overlap = Column(INTEGER, nullable=False)
    IsActive = Column(Boolean, default=True)
    ChunksDir = Column(VARCHAR(200))

    # Relationship to IndexedDocument
    Document = relationship("IndexedDocument", back_populates="ChunkGroups")

    # Relationship to IndexedDocumentChunk
    Chunks = relationship("IndexedDocumentChunk", back_populates="ChunkGroup")
