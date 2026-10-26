# Data dictionary

This project generates synthetic records. The values are illustrative and should not be interpreted as real customer behavior.

| Field | Type | Meaning | Allowed values / range |
|---|---|---|---|
| `tenure_months` | integer | Number of months as a customer | 1–72 in generated data |
| `monthly_charges` | float | Monthly bill amount in arbitrary currency units | approximately 18–150 |
| `contract_type` | category | Contract duration | `month_to_month`, `one_year`, `two_year` |
| `support_tickets` | integer | Recent support-ticket count | 0–10 in generated data |
| `payment_method` | category | Payment method | `electronic`, `bank_transfer`, `credit_card` |
| `churn` | integer target | Whether the customer churned | `0` no, `1` yes |

Missing feature values are intentionally injected into a small portion of generated records to demonstrate imputation. The target must contain both classes and only values 0/1.
