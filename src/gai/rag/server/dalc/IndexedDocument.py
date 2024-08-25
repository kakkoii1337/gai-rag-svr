from sqlalchemy import Column, PrimaryKeyConstraint, Text, VARCHAR, DateTime, Boolean, BLOB, JSON, INTEGER, Date,BIGINT
from sqlalchemy.orm import relationship
from gai.rag.server.dalc.Base import Base
from gai.rag.server.dalc.IndexedDocumentChunk import IndexedDocumentChunk

class IndexedDocument(Base):
    __tablename__ = 'IndexedDocuments'

    Id = Column(VARCHAR(44), nullable=False)
    CollectionName = Column(VARCHAR(200), nullable=False)
    ByteSize = Column(BIGINT, nullable=False)
    FileName = Column(VARCHAR(200))
    FileType = Column(VARCHAR(10))
    File = Column(BLOB)
    Source = Column(VARCHAR(255))
    Abstract = Column(Text)
    Authors = Column(VARCHAR(255))
    Title = Column(VARCHAR(255))
    Publisher = Column(VARCHAR(255))
    PublishedDate = Column(Date)
    Comments = Column(Text)
    Keywords = Column(Text)
    IsActive = Column(Boolean, default=True)
    CreatedAt = Column(DateTime)
    UpdatedAt = Column(DateTime)

    ChunkGroups = relationship("IndexedDocumentChunkGroup", back_populates="Document")

    __table_args__ = (
        PrimaryKeyConstraint('Id', 'CollectionName'),
    )