from sqladmin import ModelView

from app.models.analysis import Analysis
from app.models.user import User


class UserAdmin(ModelView, model=User):
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-users"

    column_list = [User.id, User.email, User.is_active, User.is_premium, User.created_at]
    column_searchable_list = [User.email]
    column_sortable_list = [User.id, User.created_at, User.is_premium]
    column_default_sort = [(User.created_at, True)]

    form_excluded_columns = [User.hashed_password, User.analyses]
    can_create = False
    can_delete = False


class AnalysisAdmin(ModelView, model=Analysis):
    name = "Analysis"
    name_plural = "Analyses"
    icon = "fa-solid fa-chart-line"

    column_list = [
        Analysis.id,
        Analysis.user_id,
        Analysis.platform,
        Analysis.original_filename,
        Analysis.status,
        Analysis.created_at,
    ]
    column_searchable_list = [Analysis.original_filename]
    column_sortable_list = [Analysis.id, Analysis.status, Analysis.created_at]
    column_default_sort = [(Analysis.created_at, True)]
    column_filters = [Analysis.status, Analysis.platform]

    can_create = False
    can_delete = True
