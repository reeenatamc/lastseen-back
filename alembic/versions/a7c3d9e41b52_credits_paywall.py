"""credits_paywall

Revision ID: a7c3d9e41b52
Revises: 1d2f3781db34
Create Date: 2026-10-02 00:00:00.000000

Adds user credits, the analysis unlock gate and the payments ledger.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a7c3d9e41b52'
down_revision: Union[str, None] = '1d2f3781db34'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column('credits', sa.Integer(), server_default='0', nullable=False),
    )
    op.add_column(
        'analyses',
        sa.Column('unlocked', sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    # Results that existed before the paywall stay visible
    op.execute("UPDATE analyses SET unlocked = true")
    op.add_column(
        'analyses',
        sa.Column('credits_spent', sa.Integer(), server_default='0', nullable=False),
    )

    op.create_table(
        'payments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('provider_order_id', sa.String(length=100), nullable=False),
        sa.Column('credits', sa.Integer(), nullable=False),
        sa.Column('amount_cents', sa.Integer(), nullable=True),
        sa.Column('currency', sa.String(length=10), nullable=True),
        sa.Column(
            'status',
            sa.Enum('paid', 'refunded', 'unmatched', name='paymentstatus'),
            nullable=False,
        ),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('provider', 'provider_order_id', name='uq_payments_provider_order'),
    )
    op.create_index(op.f('ix_payments_id'), 'payments', ['id'], unique=False)
    op.create_index(op.f('ix_payments_user_id'), 'payments', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_payments_user_id'), table_name='payments')
    op.drop_index(op.f('ix_payments_id'), table_name='payments')
    op.drop_table('payments')
    op.execute("DROP TYPE IF EXISTS paymentstatus")
    op.drop_column('analyses', 'credits_spent')
    op.drop_column('analyses', 'unlocked')
    op.drop_column('users', 'credits')
