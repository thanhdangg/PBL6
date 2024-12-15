import bcrypt
from sqlalchemy.orm import Session
from infra.Database.models.user import User
from infra.Database.models.prediction import Prediction
from infra.Database.schema import UserCreate, PredictionCreate

def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()
def find_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    db_user = User(username=user.username, password=hashed_password.decode('utf-8'), role=user.role)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_prediction(db: Session, prediction_id: int):
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()

def get_predictions(db: Session, skip: int = 0, limit: int = 100, user_id: int = None):
    return db.query(Prediction).filter(Prediction.user_id == user_id).offset(skip).limit(limit).all()

def create_prediction(db: Session, prediction: PredictionCreate):
    db_prediction = Prediction(
        raw_image=prediction.raw_image,
        segment_image=prediction.segment_image,
        prediction_result=prediction.prediction_result,
        user_id=prediction.user_id
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction