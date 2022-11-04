"""Main module."""

import sys

from dependency_injector.wiring import Provide, inject

from .containers import Application
from .services import AuthService, PhotoService, UserService


@inject
def train(
        email: str,
        password: str,
        photo: str,
        user_service: UserService = Provide[Application.services.user],
        auth_service: AuthService = Provide[Application.services.auth],
        photo_service: PhotoService = Provide[Application.services.photo],
) -> None:
    # NOTE. 실행되어야하는 순서
    # 1. 설정값 읽어들이기
    #   1.1. 만약 Dependency injector package때문에 필요없다면 생략 가능
    # 2. DataSet 로드하기
    # 3. DataLoader 로드하기
    # 4. 모델 로드하기
    # 5. 모델 컨테이너 로드하기
    # 6. Checkpoint 로드하기
    # 7. Logger(Performance Monitor) 로드하기
    # 8. Trainer에 몽땅 집어넣어서 학습하기
    user = user_service.get_user(email)
    auth_service.authenticate(user, password)
    photo_service.upload_photo(user, photo)


if __name__ == "__main__":
    application = Application()
    application.core.init_resources()
    application.wire(modules=[__name__])

    train(*sys.argv[1:])
