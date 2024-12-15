# infra/Database/schema.py
from pydantic import BaseModel
from typing import List, Optional
from shared.enum_role_user import User_role


class UserBase(BaseModel):
    username: str
    password: str


class UserCreate(UserBase):
    role: Optional[User_role] = User_role.USER


class User(UserBase):
    id: int
    role: User_role
    predictions: List["Prediction"] = []

    class Config:
        orm_mode = True


class PredictionBase(BaseModel):
    raw_image: str
    segment_image: str
    prediction_result: str


class PredictionCreate(BaseModel):
    raw_image: str
    segment_image: str
    prediction_result: str
    user_id: int


class Prediction(PredictionBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True
