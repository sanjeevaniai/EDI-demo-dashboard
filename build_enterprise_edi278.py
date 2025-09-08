#!/usr/bin/env python3
"""
Enterprise-Scale EDI-278 Prior Authorization Dataset Generator

This script generates hundreds of millions of synthetic EDI-278 records
for enterprise-scale healthcare analytics demonstrations.

Author: Data Engineering Team
Purpose: Enterprise-scale confidentiality-safe demo package
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import string
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Enterprise Configuration
N_ROWS = 500000000  # 500 million records for true enterprise scale
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
RANDOM_SEED = 42
BATCH_SIZE = 1000000  # 1M records per batch for efficiency

# Fortune 25 healthtech operating states (same as main script)
FORTUNE25_HEALTHTECH_STATES = {
    'AL': {'name': 'Alabama', 'plans': ['Ambetter from Alabama', 'Peach State Health Plan'], 'lob': ['Medicaid', 'Marketplace']},
    'AK': {'name': 'Alaska', 'plans': ['Alaska Total Care'], 'lob': ['Medicaid']},
    'AZ': {'name': 'Arizona', 'plans': ['Ambetter from Arizona', 'Arizona Complete Health'], 'lob': ['Medicaid', 'Marketplace']},
    'AR': {'name': 'Arkansas', 'plans': ['Ambetter from Arkansas', 'Arkansas Total Care'], 'lob': ['Medicaid', 'Marketplace']},
    'CA': {'name': 'California', 'plans': ['Health Net', 'California Health & Wellness'], 'lob': ['Medicaid', 'Medicare', 'Marketplace']},
    'CO': {'name': 'Colorado', 'plans': ['Ambetter from Colorado', 'Colorado Access'], 'lob': ['Medicaid', 'Marketplace']},
    'CT': {'name': 'Connecticut', 'plans': ['Connecticut Total Care'], 'lob': ['Medicaid']},
    'DE': {'name': 'Delaware', 'plans': ['Delaware Total Care'], 'lob': ['Medicaid']},
    'FL': {'name': 'Florida', 'plans': ['Sunshine Health', 'Ambetter from Florida'], 'lob': ['Medicaid', 'Marketplace']},
    'GA': {'name': 'Georgia', 'plans': ['Peach State Health Plan', 'Ambetter from Georgia'], 'lob': ['Medicaid', 'Marketplace']},
    'HI': {'name': 'Hawaii', 'plans': ['Hawaii Medical Service Association'], 'lob': ['Medicaid']},
    'ID': {'name': 'Idaho', 'plans': ['Idaho Total Care'], 'lob': ['Medicaid']},
    'IL': {'name': 'Illinois', 'plans': ['IlliniCare Health', 'Ambetter from Illinois'], 'lob': ['Medicaid', 'Marketplace']},
    'IN': {'name': 'Indiana', 'plans': ['MHS Indiana', 'Ambetter from Indiana'], 'lob': ['Medicaid', 'Marketplace']},
    'IA': {'name': 'Iowa', 'plans': ['Iowa Total Care'], 'lob': ['Medicaid']},
    'KS': {'name': 'Kansas', 'plans': ['Sunflower Health Plan', 'Ambetter from Kansas'], 'lob': ['Medicaid', 'Marketplace']},
    'KY': {'name': 'Kentucky', 'plans': ['WellCare of Kentucky', 'Ambetter from Kentucky'], 'lob': ['Medicaid', 'Marketplace']},
    'LA': {'name': 'Louisiana', 'plans': ['Louisiana Healthcare Connections', 'Ambetter from Louisiana'], 'lob': ['Medicaid', 'Marketplace']},
    'ME': {'name': 'Maine', 'plans': ['MaineCare'], 'lob': ['Medicaid']},
    'MD': {'name': 'Maryland', 'plans': ['Ambetter from Maryland'], 'lob': ['Marketplace']},
    'MA': {'name': 'Massachusetts', 'plans': ['Commonwealth Care Alliance'], 'lob': ['Medicaid', 'Medicare']},
    'MI': {'name': 'Michigan', 'plans': ['Meridian Health Plan', 'Ambetter from Michigan'], 'lob': ['Medicaid', 'Marketplace']},
    'MN': {'name': 'Minnesota', 'plans': ['UCare'], 'lob': ['Medicaid', 'Medicare']},
    'MS': {'name': 'Mississippi', 'plans': ['Magnolia Health Plan', 'Ambetter from Mississippi'], 'lob': ['Medicaid', 'Marketplace']},
    'MO': {'name': 'Missouri', 'plans': ['Home State Health Plan', 'Ambetter from Missouri'], 'lob': ['Medicaid', 'Marketplace']},
    'MT': {'name': 'Montana', 'plans': ['Montana Total Care'], 'lob': ['Medicaid']},
    'NE': {'name': 'Nebraska', 'plans': ['Nebraska Total Care'], 'lob': ['Medicaid']},
    'NV': {'name': 'Nevada', 'plans': ['SilverSummit Healthplan', 'Ambetter from Nevada'], 'lob': ['Medicaid', 'Marketplace']},
    'NH': {'name': 'New Hampshire', 'plans': ['New Hampshire Healthy Families'], 'lob': ['Medicaid']},
    'NJ': {'name': 'New Jersey', 'plans': ['Horizon NJ Health'], 'lob': ['Medicaid']},
    'NM': {'name': 'New Mexico', 'plans': ['Molina Healthcare of New Mexico'], 'lob': ['Medicaid']},
    'NY': {'name': 'New York', 'plans': ['Fidelis Care', 'Ambetter from New York'], 'lob': ['Medicaid', 'Marketplace']},
    'NC': {'name': 'North Carolina', 'plans': ['WellCare of North Carolina', 'Ambetter from North Carolina'], 'lob': ['Medicaid', 'Marketplace']},
    'ND': {'name': 'North Dakota', 'plans': ['North Dakota Total Care'], 'lob': ['Medicaid']},
    'OH': {'name': 'Ohio', 'plans': ['Buckeye Health Plan', 'Ambetter from Ohio'], 'lob': ['Medicaid', 'Marketplace']},
    'OK': {'name': 'Oklahoma', 'plans': ['Oklahoma Complete Health', 'Ambetter from Oklahoma'], 'lob': ['Medicaid', 'Marketplace']},
    'OR': {'name': 'Oregon', 'plans': ['Trillium Community Health Plan'], 'lob': ['Medicaid']},
    'PA': {'name': 'Pennsylvania', 'plans': ['Gateway Health Plan', 'Ambetter from Pennsylvania'], 'lob': ['Medicaid', 'Marketplace']},
    'RI': {'name': 'Rhode Island', 'plans': ['Neighborhood Health Plan of Rhode Island'], 'lob': ['Medicaid']},
    'SC': {'name': 'South Carolina', 'plans': ['Absolute Total Care', 'Ambetter from South Carolina'], 'lob': ['Medicaid', 'Marketplace']},
    'SD': {'name': 'South Dakota', 'plans': ['South Dakota Total Care'], 'lob': ['Medicaid']},
    'TN': {'name': 'Tennessee', 'plans': ['TennCare Select', 'Ambetter from Tennessee'], 'lob': ['Medicaid', 'Marketplace']},
    'TX': {'name': 'Texas', 'plans': ['Superior HealthPlan', 'Ambetter from Texas'], 'lob': ['Medicaid', 'Marketplace']},
    'UT': {'name': 'Utah', 'plans': ['Molina Healthcare of Utah'], 'lob': ['Medicaid']},
    'VT': {'name': 'Vermont', 'plans': ['Vermont Total Care'], 'lob': ['Medicaid']},
    'VA': {'name': 'Virginia', 'plans': ['Virginia Premier', 'Ambetter from Virginia'], 'lob': ['Medicaid', 'Marketplace']},
    'WA': {'name': 'Washington', 'plans': ['Coordinated Care', 'Ambetter from Washington'], 'lob': ['Medicaid', 'Marketplace']},
    'WV': {'name': 'West Virginia', 'plans': ['The Health Plan', 'Ambetter from West Virginia'], 'lob': ['Medicaid', 'Marketplace']},
    'WI': {'name': 'Wisconsin', 'plans': ['Network Health Plan'], 'lob': ['Medicaid']},
    'WY': {'name': 'Wyoming', 'plans': ['Wyoming Total Care'], 'lob': ['Medicaid']}
}

# Set random seeds
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_alphanumeric_id(prefix, length=8):
    """Generate alphanumeric ID with given prefix and length."""
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{prefix}{suffix}"

def generate_masked_npi():
    """Generate masked NPI (********1234 format)."""
    last_four = random.randint(1000, 9999)
    return f"********{last_four}"

def generate_masked_member_id():
    """Generate masked member ID (MBR + 6 digits)."""
    digits = random.randint(100000, 999999)
    return f"MBR{digits}"

def generate_timestamp(start_date, end_date):
    """Generate random timestamp between start and end dates."""
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_seconds = random.randint(0, 59)
    
    return start_date + timedelta(
        days=random_days,
        hours=random_hours,
        minutes=random_minutes,
        seconds=random_seconds
    )

def calculate_turnaround_hours(service_type, urgent_flag, final_status):
    """Calculate turnaround hours based on service complexity and other factors."""
    base_tat = max(1, int(np.random.gamma(2, 8)))
    
    complex_services = {'Surgery', 'Chemo', 'Radiation'}
    if service_type in complex_services:
        base_tat = int(base_tat * 1.3)
    
    if urgent_flag:
        base_tat = int(base_tat * 0.7)
    
    if final_status == 'denied':
        base_tat = int(base_tat * 1.2)
    
    return max(1, base_tat)

def generate_enterprise_data():
    """Generate enterprise-scale EDI-278 dataset with optimized processing."""
    print(f"Generating {N_ROWS:,} enterprise-scale EDI-278 records...")
    print(f"This will create approximately {N_ROWS * 0.0002 / 1024:.1f} GB of data")
    print(f"Estimated processing time: {N_ROWS // 1000000} minutes")
    
    # Service type weights
    service_weights = {
        'PhysicalTherapy': 0.25,
        'OccupationalTherapy': 0.10,
        'Imaging': 0.15,
        'DME': 0.10,
        'HomeHealth': 0.10,
        'Surgery': 0.08,
        'Chemo': 0.08,
        'Radiation': 0.14
    }
    
    # State distribution
    state_codes = list(FORTUNE25_HEALTHTECH_STATES.keys())
    state_weights = [0.05] * len(state_codes)
    large_states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    for state in large_states:
        if state in state_codes:
            idx = state_codes.index(state)
            state_weights[idx] = 0.08
    
    total_weight = sum(state_weights)
    state_weights = [w/total_weight for w in state_weights]
    
    # Calculate batches
    num_batches = (N_ROWS + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Processing {num_batches} batches of {BATCH_SIZE:,} records each...")
    
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, N_ROWS)
        actual_batch_size = end_idx - start_idx
        
        print(f"Processing batch {batch_num + 1}/{num_batches}: records {start_idx:,} to {end_idx:,}")
        
        # Generate batch data
        data = []
        for i in range(actual_batch_size):
            # Generate identifiers
            request_id = generate_alphanumeric_id('REQ')
            prior_auth_id = generate_alphanumeric_id('PA')
            submitter_id = generate_alphanumeric_id('SUB')
            provider_npi_masked = generate_masked_npi()
            facility_id = generate_alphanumeric_id('FAC')
            member_id_masked = generate_masked_member_id()
            
            # Generate timestamps
            request_timestamp = generate_timestamp(START_DATE, END_DATE)
            
            # Generate state and health plan
            state_code = np.random.choice(state_codes, p=state_weights)
            state_info = FORTUNE25_HEALTHTECH_STATES[state_code]
            health_plan = random.choice(state_info['plans'])
            line_of_business = random.choice(state_info['lob'])
            
            # Generate other fields
            service_type = np.random.choice(list(service_weights.keys()), p=list(service_weights.values()))
            place_of_service = np.random.choice(['Outpatient', 'Inpatient', 'Ambulatory', 'Home'], 
                                              p=[0.50, 0.20, 0.20, 0.10])
            age_band = np.random.choice(['0-17', '18-34', '35-49', '50-64', '65+'], 
                                      p=[0.05, 0.20, 0.25, 0.30, 0.20])
            request_type = np.random.choice(['initial', 'concurrent', 'extension'], 
                                          p=[0.70, 0.20, 0.10])
            
            # Generate flags
            urgent_flag = 1 if random.random() < 0.15 else 0
            resubmission_flag = 1 if random.random() < 0.10 else 0
            
            # Generate status
            initial_status = np.random.choice(['received', 'pended', 'approved', 'denied'], 
                                            p=[0.05, 0.15, 0.70, 0.10])
            
            if initial_status == 'pended':
                final_status = 'approved' if random.random() < 0.80 else 'denied' if random.random() < 0.60 else 'pended'
            else:
                final_status = initial_status
            
            # Calculate turnaround time
            turnaround_hours = calculate_turnaround_hours(service_type, urgent_flag, final_status)
            decision_timestamp = request_timestamp + timedelta(hours=turnaround_hours)
            
            # Calculate SLA compliance
            sla_threshold = 24 if urgent_flag else 72
            sla_met = 1 if turnaround_hours <= sla_threshold else 0
            
            # Generate reasons
            pend_reason = None
            if final_status == 'pended':
                pend_reason = np.random.choice(['ClinicalInfoMissing', 'EligibilityCheck', 
                                              'CodingClarification', 'MedicalNecessityReview', 'Other'],
                                             p=[0.45, 0.15, 0.15, 0.15, 0.10])
            
            deny_reason = None
            if final_status == 'denied':
                deny_reason = np.random.choice(['NotMedicallyNecessary', 'OutOfNetwork', 
                                              'CoverageExclusion', 'NoReferral', 'Other'],
                                             p=[0.40, 0.20, 0.15, 0.15, 0.10])
            
            # Generate appeal flag
            appeal_flag = 0
            if final_status == 'denied':
                appeal_flag = 1 if random.random() < 0.20 else 0
            else:
                appeal_flag = 1 if random.random() < 0.05 else 0
            
            data.append({
                'request_id': request_id,
                'prior_auth_id': prior_auth_id,
                'submitter_id': submitter_id,
                'provider_npi_masked': provider_npi_masked,
                'facility_id': facility_id,
                'member_id_masked': member_id_masked,
                'request_timestamp': request_timestamp,
                'decision_timestamp': decision_timestamp,
                'turnaround_hours': turnaround_hours,
                'sla_met': sla_met,
                'request_type': request_type,
                'service_type': service_type,
                'place_of_service': place_of_service,
                'state_code': state_code,
                'state_name': state_info['name'],
                'health_plan': health_plan,
                'line_of_business': line_of_business,
                'age_band': age_band,
                'urgent_flag': urgent_flag,
                'status': final_status,
                'pend_reason': pend_reason,
                'deny_reason': deny_reason,
                'resubmission_flag': resubmission_flag,
                'appeal_flag': appeal_flag
            })
        
        # Save batch to CSV
        batch_df = pd.DataFrame(data)
        filename = f'edi278_enterprise_batch_{batch_num + 1:03d}.csv'
        batch_df.to_csv(filename, index=False)
        
        print(f"Saved batch {batch_num + 1} to {filename} ({len(data):,} records)")
        
        # Memory cleanup
        del data, batch_df
    
    print(f"\nEnterprise dataset generation complete!")
    print(f"Generated {num_batches} batch files with {N_ROWS:,} total records")
    print(f"Total estimated size: {N_ROWS * 0.0002 / 1024:.1f} GB")

if __name__ == "__main__":
    print("="*80)
    print("ENTERPRISE-SCALE EDI-278 DATASET GENERATOR")
    print("="*80)
    print(f"Target Records: {N_ROWS:,}")
    print(f"Time Range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Batch Size: {BATCH_SIZE:,}")
    print(f"Estimated Processing Time: {N_ROWS // 1000000} minutes")
    print(f"Estimated Storage: {N_ROWS * 0.0002 / 1024:.1f} GB")
    print("="*80)
    
    response = input("This will generate 500 million records. Continue? (y/N): ")
    if response.lower() == 'y':
        generate_enterprise_data()
    else:
        print("Operation cancelled.")
