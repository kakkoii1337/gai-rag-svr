from sqlalchemy import Column, PrimaryKeyConstraint, Text, VARCHAR, DateTime, Boolean, BLOB, JSON, INTEGER, Date,BIGINT
from sqlalchemy.orm import relationship
from gai.rag.server.dalc.Base import Base
from gai.rag.server.dalc.IndexedDocumentChunk import IndexedDocumentChunk

from gai.rag.server.dtos.indexed_doc import IndexedDocPydantic

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

    def to_pydantic(self):
        try:
            return IndexedDocPydantic(**{
                'Id': self.Id,
                'CollectionName': self.CollectionName,
                'ByteSize': self.ByteSize,
                'FileName': self.FileName,
                'FileType': self.FileType,
                'File': self.File,
                'Source': self.Source,
                'Abstract': self.Abstract,
                'Authors': self.Authors,
                'Title': self.Title,
                'Publisher': self.Publisher,
                'PublishedDate': self.PublishedDate,
                'Comments': self.Comments,
                'Keywords': self.Keywords,
                'IsActive': self.IsActive,
                'CreatedAt': self.CreatedAt,
                'UpdatedAt': self.UpdatedAt,
                'ChunkGroups': [cg.to_pydantic() for cg in self.ChunkGroups]
            })
        except Exception as e:
            raise RuntimeError("Failed to convert from DALC to Pydantic model.") from e