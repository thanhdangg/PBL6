# infra/Database/models/user.py
from sqlmodel import SQLModel, Field, Relationship
from shared.enum_role_user import User_role


class User(SQLModel, table=True):
    __tablename__ = "User"
    id: int | None = Field(default=None, primary_key=True)
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)
    role: User_role | None = Field(default=User_role.USER)
    predictions: list["Prediction"] = Relationship(back_populates="user")
