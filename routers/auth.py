from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from infra
from app.repositories.user_repo import create_user, get_user_by_email
from app.schemas.user_schema import UserCreate, UserResponse
from app.utils.password_hash import verify_password
from app.utils.jwt_handler import create_access_token
from datetime import timedelta

router = APIRouter()
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_user(db, user)

@router.post("/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, user.email)
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(
        {"sub": db_user.email}, expires_delta=timedelta(hours=1)
    )
    return {"access_token": access_token, "token_type": "bearer"}
