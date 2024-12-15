from sqlalchemy.orm import Session
from infra.Database.models.user import User
from infra.Database.schema import UserCreate
from utils.password_hash import hash_password
def create_user(db: Session, user: UserCreate):
    db_user = User(
        username=user.username,
        email=user.email,
        password=hash_password(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_email(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()
