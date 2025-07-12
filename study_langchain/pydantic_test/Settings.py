from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "MyApp"
    admin_email: Optional[str] = Field(None, description='邮件地址')

    class Config:
        env_file = ".env"  # 从 .env 文件加载环境变量


# Pydantic 提供了 BaseSettings 类，支持从环境变量中读取配置：
settings = Settings()
print(settings.admin_email)
