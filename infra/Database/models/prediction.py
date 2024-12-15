# infra/Database/models/prediction.py
from sqlmodel import SQLModel, Field, Relationship


class Prediction(SQLModel, table=True):
    __tablename__ = "Prediction"
    id: int | None = Field(default=None, primary_key=True)
    raw_image: str | None = Field(default=None)
    segment_image: str | None = Field(default=None)
    prediction_result: str | None = Field(default=None)
    user_id: int | None = Field(default=None, foreign_key="User.id")
    user: "User" = Relationship(back_populates="predictions")
