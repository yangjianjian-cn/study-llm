from pydantic import root_validator, BaseModel, model_validator


class Rectangle(BaseModel):
    width: int
    height: int
    area: int = 0

    # 跨字段验证
    @model_validator
    def calculate_area(cls, values):
        width = values.get('width')
        height = values.get('height')
        values['area'] = width * height
        return values
