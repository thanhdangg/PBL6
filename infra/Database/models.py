from sqlmodel import SQLModel, Field, Relationship
from shared.enum_role_user import User_role
from database import Base


class Prediction(Base):
    __tablename__ = "Prediction"
    id: int | None = Field(default=None, primary_key=True)
    raw_image: str | None = Field(default=None)
    segment_image: str | None = Field(default=None)
    prediction_result: str | None = Field(default=None)
    user_id: int | None = Field(default=None, foreign_key="User.id")
    user: "User" = Relationship(back_populates="predictions")


class User(Base):
    __tablename__ = "User"
    id: int | None = Field(default=None, primary_key=True)
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)
    role: User_role | None = Field(default=User_role.USER)
    predictions: list["Prediction"] = Relationship(back_populates="user")
