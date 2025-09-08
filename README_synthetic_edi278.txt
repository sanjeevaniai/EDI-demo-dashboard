SYNTHETIC EDI-278 PRIOR AUTHORIZATION DATASET
=====================================================

IMPORTANT NOTICE:
This dataset contains 100% synthetic data generated for demonstration purposes only.
No real patient information, PII, or PHI is included. This data is solely for
demonstrative, non-production use and should never be used for actual healthcare
operations or decision-making.

DATASET OVERVIEW:
- Total Records: 600
- Date Range: 2024-06-01 to 2024-10-31
- All data is completely synthetic and randomly generated

FIELD DESCRIPTIONS:
==================

Identifiers:
- request_id: Unique request identifier (REQ + 8 alphanumeric)
- prior_auth_id: Prior authorization ID (PA + 8 alphanumeric)
- submitter_id: Submitting entity ID (SUB + 8 alphanumeric)
- provider_npi_masked: Masked provider NPI (********1234 format)
- facility_id: Facility identifier (FAC + 8 alphanumeric)
- member_id_masked: Masked member ID (MBR + 6 digits)

Timestamps:
- request_timestamp: When the prior auth request was submitted (ISO format)
- decision_timestamp: When the decision was made (ISO format)
- turnaround_hours: Hours between request and decision (integer)

SLA & Status:
- sla_met: Whether SLA was met (1=yes, 0=no)
- status: Final status (received, pended, approved, denied)
- urgent_flag: Whether request was marked urgent (1=yes, 0=no)

Request Details:
- request_type: Type of request (initial, concurrent, extension)
- service_type: Type of service requested
- place_of_service: Where service will be provided
- state_code: US state code (2-letter)
- state_name: Full state name
- health_plan: Fortune 25 healthtech plan name for the state
- line_of_business: Medicaid, Medicare, or Marketplace
- age_band: Patient age group

Resolution Details:
- pend_reason: Reason for pended status (if applicable)
- deny_reason: Reason for denial (if applicable)
- resubmission_flag: Whether this is a resubmission (1=yes, 0=no)
- appeal_flag: Whether an appeal was filed (1=yes, 0=no)

SLA DEFINITION:
- Urgent requests: 24-hour SLA
- Non-urgent requests: 72-hour SLA

SUGGESTED DASHBOARD METRICS:
============================

Volume & Trends:
- Daily/weekly request volume trends
- Request volume by service type, state, age band
- Resubmission and appeal rates

Status Analysis:
- Status distribution and trends over time
- Pended request resolution rates and timing
- Denial rate trends and patterns

Performance Metrics:
- Turnaround time distribution and trends
- SLA compliance rates by urgency level
- Performance by service type and complexity

Operational Insights:
- Top denial reasons and trends
- Pend reason analysis
- Geographic and demographic breakdowns
- Provider and facility performance

Data Quality:
- Missing data analysis
- Data completeness by field
- Anomaly detection in turnaround times

MASKING NOTES:
- Provider NPIs are masked as ********1234 format
- Member IDs are masked as MBR + 6 digits
- All other identifiers are synthetic and randomly generated

GENERATION ASSUMPTIONS:
======================

Distribution Assumptions:
- 70% initial requests, 20% concurrent, 10% extensions
- 15% urgent requests
- 70% approved, 10% denied, 15% pended, 5% received
- 60% of pended requests resolve to approved/denied
- 10% resubmission rate
- 20% appeal rate for denied requests, 5% for others

Service Type Distribution:
- Physical Therapy: 25%
- Imaging: 15%
- Radiation: 14%
- Other services: 8-10% each

Turnaround Time Logic:
- Base TAT: Gamma distribution (right-skewed)
- Complex services (Surgery, Chemo, Radiation): +30% TAT
- Urgent requests: -30% TAT
- Denied requests: +20% TAT

This synthetic dataset provides a realistic foundation for demonstrating
healthcare prior authorization dashboard capabilities while maintaining
complete confidentiality and data safety.
