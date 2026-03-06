from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    fireworks_api_key:      str
    anthropic_api_key:      str
    wandb_api_key:          str
    fireworks_account_id:   str

    sft_train_dataset_id:   str = ''
    sft_val_dataset_id:     str = ''
    dpo_train_dataset_id:   str = ''
    dpo_val_dataset_id:     str = ''
    sft_model_id:           str = ''
    dpo_model_id:           str = ''

    base_model: str = 'accounts/fireworks/models/qwen2p5-7b-instruct'

    required_fields: list[str] = [
        'title', 'company', 'location', 'remote',
        'experience_years', 'salary_min', 'salary_max',
        'skills', 'deadline', 'benefits']

    class Config:
        env_file = '.env'

settings = Settings()
