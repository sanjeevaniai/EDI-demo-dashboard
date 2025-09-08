#!/usr/bin/env python3
"""
Fix dashboard data and create corrected interactive dashboards
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

def clean_and_fix_data():
    """Clean the CSV data and create properly formatted JSON for dashboards."""
    
    print("Loading and cleaning data...")
    
    # Read the CSV and clean it
    df = pd.read_csv('synthetic_edi278_dataset_sample.csv')
    
    # Clean the data
    df = df.dropna()  # Remove any rows with NaN values
    
    # Convert timestamps properly
    df['request_timestamp'] = pd.to_datetime(df['request_timestamp'])
    df['decision_timestamp'] = pd.to_datetime(df['decision_timestamp'])
    
    # Ensure numeric columns are properly formatted
    df['turnaround_hours'] = pd.to_numeric(df['turnaround_hours'], errors='coerce')
    df['sla_met'] = pd.to_numeric(df['sla_met'], errors='coerce')
    df['urgent_flag'] = pd.to_numeric(df['urgent_flag'], errors='coerce')
    df['resubmission_flag'] = pd.to_numeric(df['resubmission_flag'], errors='coerce')
    df['appeal_flag'] = pd.to_numeric(df['appeal_flag'], errors='coerce')
    
    # Remove any rows that couldn't be converted
    df = df.dropna()
    
    print(f"Cleaned data: {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    # Convert to JSON with proper formatting
    data_json = df.to_json(orient='records', date_format='iso')
    
    # Get unique values for filters
    unique_states = sorted(df['state_name'].unique().tolist())
    unique_services = sorted(df['service_type'].unique().tolist())
    unique_health_plans = sorted(df['health_plan'].unique().tolist())
    unique_lob = sorted(df['line_of_business'].unique().tolist())
    unique_statuses = sorted(df['status'].unique().tolist())
    
    # Date range
    min_date = df['request_timestamp'].min().strftime('%Y-%m-%d')
    max_date = df['request_timestamp'].max().strftime('%Y-%m-%d')
    
    print(f"Date range: {min_date} to {max_date}")
    print(f"Unique states: {len(unique_states)}")
    print(f"Unique services: {len(unique_services)}")
    
    return data_json, unique_states, unique_services, unique_health_plans, unique_lob, unique_statuses, min_date, max_date

def create_fixed_interactive_dashboard():
    """Create a fixed interactive dashboard with proper data handling."""
    
    data_json, unique_states, unique_services, unique_health_plans, unique_lob, unique_statuses, min_date, max_date = clean_and_fix_data()
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDI-278 Interactive Dashboard - Fixed</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin: 10px 0 0 0;
        }}
        
        .filters-section {{
            background: #f8f9fa;
            padding: 25px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .filters-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .filter-group {{
            display: flex;
            flex-direction: column;
        }}
        
        .filter-group label {{
            font-weight: 600;
            margin-bottom: 8px;
            color: #2c3e50;
            font-size: 0.9em;
        }}
        
        .filter-group select, .filter-group input {{
            padding: 10px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
        }}
        
        .filter-group select:focus, .filter-group input:focus {{
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }}
        
        .action-buttons {{
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }}
        
        .btn-secondary {{
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
            color: white;
        }}
        
        .btn-secondary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(149, 165, 166, 0.4);
        }}
        
        .kpi-section {{
            padding: 30px;
            background: white;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .kpi-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-5px);
        }}
        
        .kpi-card h3 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .kpi-card p {{
            font-size: 1.1em;
            opacity: 0.9;
            margin: 0;
        }}
        
        .charts-section {{
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
        }}
        
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .chart-container h2 {{
            color: #34495e;
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        
        .chart-container canvas {{
            width: 100%;
            height: 400px;
        }}
        
        .data-table {{
            margin-top: 30px;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .data-table h3 {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            margin: 0;
        }}
        
        .table-container {{
            max-height: 400px;
            overflow-y: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            position: sticky;
            top: 0;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .status-approved {{ color: #27ae60; font-weight: 600; }}
        .status-denied {{ color: #e74c3c; font-weight: 600; }}
        .status-pended {{ color: #f39c12; font-weight: 600; }}
        .status-received {{ color: #3498db; font-weight: 600; }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .footer p {{
            margin: 5px 0;
            opacity: 0.8;
        }}
        
        .loading {{
            text-align: center;
            padding: 50px;
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .error {{
            color: #e74c3c;
            background: #fdf2f2;
            padding: 15px;
            border-radius: 8px;
            margin: 20px;
            border-left: 4px solid #e74c3c;
        }}
        
        @media (max-width: 768px) {{
            .filters-grid {{
                grid-template-columns: 1fr;
            }}
            
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .kpi-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>EDI-278 Interactive Dashboard</h1>
            <p>Real-time Analytics & Performance Monitoring</p>
        </div>
        
        <div class="filters-section">
            <div class="filters-grid">
                <div class="filter-group">
                    <label for="dateFrom">From Date</label>
                    <input type="date" id="dateFrom" value="{min_date}">
                </div>
                <div class="filter-group">
                    <label for="dateTo">To Date</label>
                    <input type="date" id="dateTo" value="{max_date}">
                </div>
                <div class="filter-group">
                    <label for="stateFilter">State</label>
                    <select id="stateFilter">
                        <option value="">All States</option>
                        {''.join([f'<option value="{state}">{state}</option>' for state in unique_states])}
                    </select>
                </div>
                <div class="filter-group">
                    <label for="serviceFilter">Service Type</label>
                    <select id="serviceFilter">
                        <option value="">All Services</option>
                        {''.join([f'<option value="{service}">{service}</option>' for service in unique_services])}
                    </select>
                </div>
                <div class="filter-group">
                    <label for="healthPlanFilter">Health Plan</label>
                    <select id="healthPlanFilter">
                        <option value="">All Health Plans</option>
                        {''.join([f'<option value="{plan}">{plan}</option>' for plan in unique_health_plans])}
                    </select>
                </div>
                <div class="filter-group">
                    <label for="lobFilter">Line of Business</label>
                    <select id="lobFilter">
                        <option value="">All LOB</option>
                        {''.join([f'<option value="{lob}">{lob}</option>' for lob in unique_lob])}
                    </select>
                </div>
                <div class="filter-group">
                    <label for="statusFilter">Status</label>
                    <select id="statusFilter">
                        <option value="">All Statuses</option>
                        {''.join([f'<option value="{status}">{status}</option>' for status in unique_statuses])}
                    </select>
                </div>
                <div class="filter-group">
                    <label for="urgentFilter">Urgency</label>
                    <select id="urgentFilter">
                        <option value="">All</option>
                        <option value="1">Urgent Only</option>
                        <option value="0">Non-Urgent Only</option>
                    </select>
                </div>
            </div>
            
            <div class="action-buttons">
                <button class="btn btn-primary" onclick="applyFilters()">Apply Filters</button>
                <button class="btn btn-secondary" onclick="resetFilters()">Reset All</button>
                <button class="btn btn-secondary" onclick="exportData()">Export Data</button>
            </div>
        </div>
        
        <div class="kpi-section">
            <div class="kpi-grid">
                <div class="kpi-card">
                    <h3 id="totalRequests">-</h3>
                    <p>Total Requests</p>
                </div>
                <div class="kpi-card">
                    <h3 id="approvalRate">-</h3>
                    <p>Approval Rate</p>
                </div>
                <div class="kpi-card">
                    <h3 id="slaCompliance">-</h3>
                    <p>SLA Compliance</p>
                </div>
                <div class="kpi-card">
                    <h3 id="avgTat">-</h3>
                    <p>Avg Turnaround Time</p>
                </div>
                <div class="kpi-card">
                    <h3 id="urgentRequests">-</h3>
                    <p>Urgent Requests</p>
                </div>
                <div class="kpi-card">
                    <h3 id="appealRate">-</h3>
                    <p>Appeal Rate</p>
                </div>
            </div>
        </div>
        
        <div class="charts-section">
            <div class="charts-grid">
                <div class="chart-container">
                    <h2>Status Distribution</h2>
                    <canvas id="statusChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h2>SLA Compliance by Urgency</h2>
                    <canvas id="slaChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h2>Daily Volume Trend</h2>
                    <canvas id="volumeChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h2>Service Type Distribution</h2>
                    <canvas id="serviceChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h2>Geographic Performance (Top 10 States)</h2>
                    <canvas id="geoChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h2>Turnaround Time Distribution</h2>
                    <canvas id="tatChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="data-table">
            <h3>Filtered Data Table</h3>
            <div class="table-container">
                <table id="dataTable">
                    <thead>
                        <tr>
                            <th>Request ID</th>
                            <th>Date</th>
                            <th>State</th>
                            <th>Service Type</th>
                            <th>Status</th>
                            <th>TAT (Hours)</th>
                            <th>SLA Met</th>
                            <th>Urgent</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                        <tr><td colspan="8" class="loading">Loading data...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Interactive EDI-278 Dashboard</strong> - Real-time Analytics Platform</p>
            <p>Dynamic filtering and analysis capabilities for leadership insights</p>
            <p><em>Data: 100% synthetic for demonstration purposes</em></p>
        </div>
    </div>

    <script>
        // Global data storage
        let allData = [];
        let filteredData = [];
        
        // Chart instances
        let charts = {{}};
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Dashboard initializing...');
            
            try {{
                // Parse the JSON data
                allData = {data_json};
                filteredData = allData;
                
                console.log('Data loaded successfully:', allData.length, 'records');
                console.log('Sample record:', allData[0]);
                
                // Initialize the dashboard
                applyFilters();
                
            }} catch (error) {{
                console.error('Error loading data:', error);
                showError('Error loading data: ' + error.message);
            }}
        }});
        
        function showError(message) {{
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            document.body.insertBefore(errorDiv, document.body.firstChild);
        }}
        
        // Filter functions
        function applyFilters() {{
            try {{
                const dateFrom = document.getElementById('dateFrom').value;
                const dateTo = document.getElementById('dateTo').value;
                const state = document.getElementById('stateFilter').value;
                const service = document.getElementById('serviceFilter').value;
                const healthPlan = document.getElementById('healthPlanFilter').value;
                const lob = document.getElementById('lobFilter').value;
                const status = document.getElementById('statusFilter').value;
                const urgent = document.getElementById('urgentFilter').value;
                
                filteredData = allData.filter(record => {{
                    const recordDate = new Date(record.request_timestamp);
                    const fromDate = dateFrom ? new Date(dateFrom) : new Date('1900-01-01');
                    const toDate = dateTo ? new Date(dateTo) : new Date('2100-01-01');
                    
                    return (
                        recordDate >= fromDate &&
                        recordDate <= toDate &&
                        (!state || record.state_name === state) &&
                        (!service || record.service_type === service) &&
                        (!healthPlan || record.health_plan === healthPlan) &&
                        (!lob || record.line_of_business === lob) &&
                        (!status || record.status === status) &&
                        (urgent === '' || record.urgent_flag.toString() === urgent)
                    );
                }});
                
                console.log('Filtered to', filteredData.length, 'records');
                updateKPIs();
                updateCharts();
                updateTable();
                
            }} catch (error) {{
                console.error('Error applying filters:', error);
                showError('Error applying filters: ' + error.message);
            }}
        }}
        
        function resetFilters() {{
            document.getElementById('dateFrom').value = '{min_date}';
            document.getElementById('dateTo').value = '{max_date}';
            document.getElementById('stateFilter').value = '';
            document.getElementById('serviceFilter').value = '';
            document.getElementById('healthPlanFilter').value = '';
            document.getElementById('lobFilter').value = '';
            document.getElementById('statusFilter').value = '';
            document.getElementById('urgentFilter').value = '';
            applyFilters();
        }}
        
        function exportData() {{
            try {{
                const csv = convertToCSV(filteredData);
                const blob = new Blob([csv], {{ type: 'text/csv' }});
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'edi278_filtered_data.csv';
                a.click();
                window.URL.revokeObjectURL(url);
            }} catch (error) {{
                console.error('Error exporting data:', error);
                showError('Error exporting data: ' + error.message);
            }}
        }}
        
        function convertToCSV(data) {{
            if (data.length === 0) return '';
            
            const headers = Object.keys(data[0]);
            const csvContent = [
                headers.join(','),
                ...data.map(row => headers.map(header => {{
                    const value = row[header];
                    if (typeof value === 'string' && value.includes(',')) {{
                        return `"${{value}}"`;
                    }}
                    return value;
                }}).join(','))
            ].join('\\n');
            
            return csvContent;
        }}
        
        // KPI Updates
        function updateKPIs() {{
            try {{
                const total = filteredData.length;
                const approved = filteredData.filter(r => r.status === 'approved').length;
                const slaMet = filteredData.filter(r => r.sla_met === 1).length;
                const urgent = filteredData.filter(r => r.urgent_flag === 1).length;
                const appeals = filteredData.filter(r => r.appeal_flag === 1).length;
                
                const avgTat = filteredData.length > 0 
                    ? (filteredData.reduce((sum, r) => sum + parseFloat(r.turnaround_hours), 0) / filteredData.length).toFixed(1)
                    : 0;
                
                document.getElementById('totalRequests').textContent = total.toLocaleString();
                document.getElementById('approvalRate').textContent = total > 0 ? ((approved / total) * 100).toFixed(1) + '%' : '0%';
                document.getElementById('slaCompliance').textContent = total > 0 ? ((slaMet / total) * 100).toFixed(1) + '%' : '0%';
                document.getElementById('avgTat').textContent = avgTat + 'h';
                document.getElementById('urgentRequests').textContent = urgent.toLocaleString();
                document.getElementById('appealRate').textContent = total > 0 ? ((appeals / total) * 100).toFixed(1) + '%' : '0%';
                
            }} catch (error) {{
                console.error('Error updating KPIs:', error);
                showError('Error updating KPIs: ' + error.message);
            }}
        }}
        
        // Chart Updates
        function updateCharts() {{
            try {{
                updateStatusChart();
                updateSLAChart();
                updateVolumeChart();
                updateServiceChart();
                updateGeoChart();
                updateTATChart();
            }} catch (error) {{
                console.error('Error updating charts:', error);
                showError('Error updating charts: ' + error.message);
            }}
        }}
        
        function updateStatusChart() {{
            const ctx = document.getElementById('statusChart').getContext('2d');
            if (charts.statusChart) charts.statusChart.destroy();
            
            const statusCounts = {{}};
            filteredData.forEach(record => {{
                statusCounts[record.status] = (statusCounts[record.status] || 0) + 1;
            }});
            
            charts.statusChart = new Chart(ctx, {{
                type: 'doughnut',
                data: {{
                    labels: Object.keys(statusCounts),
                    datasets: [{{
                        data: Object.values(statusCounts),
                        backgroundColor: ['#27ae60', '#e74c3c', '#f39c12', '#3498db'],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
        }}
        
        function updateSLAChart() {{
            const ctx = document.getElementById('slaChart').getContext('2d');
            if (charts.slaChart) charts.slaChart.destroy();
            
            const urgentData = filteredData.filter(r => r.urgent_flag === 1);
            const nonUrgentData = filteredData.filter(r => r.urgent_flag === 0);
            
            const urgentSLA = urgentData.length > 0 ? (urgentData.filter(r => r.sla_met === 1).length / urgentData.length) * 100 : 0;
            const nonUrgentSLA = nonUrgentData.length > 0 ? (nonUrgentData.filter(r => r.sla_met === 1).length / nonUrgentData.length) * 100 : 0;
            
            charts.slaChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: ['Urgent (24h)', 'Non-Urgent (72h)'],
                    datasets: [{{
                        label: 'SLA Compliance %',
                        data: [urgentSLA, nonUrgentSLA],
                        backgroundColor: ['#e74c3c', '#3498db'],
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100
                        }}
                    }}
                }}
            }});
        }}
        
        function updateVolumeChart() {{
            const ctx = document.getElementById('volumeChart').getContext('2d');
            if (charts.volumeChart) charts.volumeChart.destroy();
            
            // Group by date
            const dailyVolume = {{}};
            filteredData.forEach(record => {{
                const date = new Date(record.request_timestamp).toISOString().split('T')[0];
                dailyVolume[date] = (dailyVolume[date] || 0) + 1;
            }});
            
            const sortedDates = Object.keys(dailyVolume).sort();
            const volumes = sortedDates.map(date => dailyVolume[date]);
            
            charts.volumeChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: sortedDates,
                    datasets: [{{
                        label: 'Daily Volume',
                        data: volumes,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{
                                unit: 'day'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function updateServiceChart() {{
            const ctx = document.getElementById('serviceChart').getContext('2d');
            if (charts.serviceChart) charts.serviceChart.destroy();
            
            const serviceCounts = {{}};
            filteredData.forEach(record => {{
                serviceCounts[record.service_type] = (serviceCounts[record.service_type] || 0) + 1;
            }});
            
            charts.serviceChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: Object.keys(serviceCounts),
                    datasets: [{{
                        label: 'Request Count',
                        data: Object.values(serviceCounts),
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y'
                }}
            }});
        }}
        
        function updateGeoChart() {{
            const ctx = document.getElementById('geoChart').getContext('2d');
            if (charts.geoChart) charts.geoChart.destroy();
            
            const stateCounts = {{}};
            filteredData.forEach(record => {{
                stateCounts[record.state_name] = (stateCounts[record.state_name] || 0) + 1;
            }});
            
            const sortedStates = Object.entries(stateCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10);
            
            charts.geoChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: sortedStates.map(s => s[0]),
                    datasets: [{{
                        label: 'Request Count',
                        data: sortedStates.map(s => s[1]),
                        backgroundColor: 'rgba(52, 152, 219, 0.8)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            ticks: {{
                                maxRotation: 45
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function updateTATChart() {{
            const ctx = document.getElementById('tatChart').getContext('2d');
            if (charts.tatChart) charts.tatChart.destroy();
            
            const tatValues = filteredData.map(r => parseFloat(r.turnaround_hours));
            const bins = 20;
            const maxTat = Math.max(...tatValues);
            const binSize = maxTat / bins;
            
            const histogram = new Array(bins).fill(0);
            tatValues.forEach(tat => {{
                const binIndex = Math.min(Math.floor(tat / binSize), bins - 1);
                histogram[binIndex]++;
            }});
            
            const labels = Array.from({{length: bins}}, (_, i) => 
                Math.round(i * binSize) + '-' + Math.round((i + 1) * binSize) + 'h'
            );
            
            charts.tatChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Frequency',
                        data: histogram,
                        backgroundColor: 'rgba(155, 89, 182, 0.8)',
                        borderColor: 'rgba(155, 89, 182, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false
                }}
            }});
        }}
        
        function updateTable() {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            
            const displayData = filteredData.slice(0, 100); // Show first 100 records
            
            if (displayData.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="8" class="loading">No data matches the current filters</td></tr>';
                return;
            }}
            
            displayData.forEach(record => {{
                const row = document.createElement('tr');
                const date = new Date(record.request_timestamp).toLocaleDateString();
                const statusClass = 'status-' + record.status.toLowerCase();
                
                row.innerHTML = `
                    <td>${{record.request_id}}</td>
                    <td>${{date}}</td>
                    <td>${{record.state_name}}</td>
                    <td>${{record.service_type}}</td>
                    <td class="${{statusClass}}">${{record.status}}</td>
                    <td>${{record.turnaround_hours}}</td>
                    <td>${{record.sla_met ? 'Yes' : 'No'}}</td>
                    <td>${{record.urgent_flag ? 'Yes' : 'No'}}</td>
                `;
                tbody.appendChild(row);
            }});
            
            if (filteredData.length > 100) {{
                const row = document.createElement('tr');
                row.innerHTML = `<td colspan="8" class="loading">Showing first 100 of ${{filteredData.length}} records</td>`;
                tbody.appendChild(row);
            }}
        }}
    </script>
</body>
</html>"""
    
    # Save the fixed dashboard
    with open('edi278_fixed_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Fixed dashboard created: edi278_fixed_dashboard.html")
    print("This version includes:")
    print("- Proper data cleaning and validation")
    print("- Error handling for data loading issues")
    print("- Better debugging information")
    print("- Fixed chart rendering")

if __name__ == "__main__":
    create_fixed_interactive_dashboard()
