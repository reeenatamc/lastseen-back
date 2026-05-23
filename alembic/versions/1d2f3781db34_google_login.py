"""google_login

Revision ID: 1d2f3781db34
Revises: e2065b1a9129
Create Date: 2026-05-14 00:00:00.000000

Adds google_sub column for Google OAuth login and makes hashed_password
nullable so Google-only accounts can exist without a local password.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '1d2f3781db34'
down_revision: Union[str, None] = 'e2065b1a9129'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column('google_sub', sa.String(length=255), nullable=True),
    )
    op.create_index(
        op.f('ix_users_google_sub'), 'users', ['google_sub'], unique=True
    )
    op.alter_column(
        'users', 'hashed_password',
        existing_type=sa.String(length=255),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        'users', 'hashed_password',
        existing_type=sa.String(length=255),
        nullable=False,
    )
    op.drop_index(op.f('ix_users_google_sub'), table_name='users')
    op.drop_column('users', 'google_sub')
