# EDI-278 Synthetic Dataset Generator & Interactive Dashboards

A comprehensive healthcare prior-authorization (EDI-278) synthetic data generator with Power BI-style interactive dashboards for executive presentations and technical demonstrations.

## 🚀 Features

### **Enterprise-Scale Data Generation**
- **50+ Million Records**: Scalable synthetic dataset generation
- **All 50 US States**: Complete Fortune 25 healthtech plan coverage
- **Realistic Business Logic**: SLA compliance, TAT calculations, status resolution
- **Privacy-Safe**: Fully synthetic data with masked identifiers

### **Interactive Dashboards**
- **Power BI-Style Filtering**: Date range, state, service type, health plan, LOB
- **Real-Time KPIs**: Dynamic metrics that update with filters
- **Executive Presentation**: Drill-down capabilities for leadership
- **Mobile Responsive**: Works on tablets and mobile devices

### **Advanced Analytics**
- **Time Series Analysis**: Daily, hourly, monthly trends
- **Machine Learning**: Regression models for TAT prediction
- **Statistical Modeling**: Correlation analysis, performance metrics
- **Geographic Analytics**: State-by-state performance comparison

## 📊 Dashboard Types

| Dashboard | Purpose | Features |
|-----------|---------|----------|
| **Working Dashboard** | `edi278_working_dashboard.html` | All charts working, basic filtering |
| **Interactive Dashboard** | `edi278_interactive_dashboard.html` | Power BI-style, advanced filtering |
| **Executive Dashboard** | `edi278_executive_dashboard.html` | Leadership presentation, drill-down |

## 🛠️ Quick Start

### **1. Generate Sample Data**
```bash
python build_synthetic_edi278.py
```
This creates:
- `synthetic_edi278_dataset_sample.csv` (100K records)
- Static PNG charts
- Documentation files

### **2. Create Interactive Dashboards**
```bash
python create_working_dashboard.py
```
This generates the working interactive dashboard.

### **3. View Dashboards**
Open any `.html` file in your browser:
- **Chrome/Edge**: Best compatibility
- **Firefox**: Good support
- **Safari**: Basic functionality

## 📈 Sample Data Schema

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | String | Unique request identifier |
| `member_id` | String | Masked member ID |
| `provider_npi` | String | Masked provider NPI |
| `state_code` | String | 2-letter state code |
| `state_name` | String | Full state name |
| `health_plan` | String | Fortune 25 healthtech plan name |
| `line_of_business` | String | Medicaid/Medicare/Marketplace |
| `service_type` | String | Medical service category |
| `request_timestamp` | DateTime | Request submission time |
| `status` | String | Approved/Denied/Pended |
| `tat_hours` | Float | Turnaround time in hours |
| `sla_compliant` | Boolean | SLA compliance status |
| `urgency_level` | String | Routine/Urgent/Stat |
| `age_band` | String | Patient age category |
| `deny_reason` | String | Denial reason (if denied) |
| `pend_reason` | String | Pending reason (if pended) |

## 🎯 Key Metrics

### **SLA Compliance**
- **Target**: 95% compliance rate
- **Calculation**: TAT ≤ SLA threshold
- **SLA Rules**: Service type + urgency level based

### **Turnaround Time (TAT)**
- **Routine**: 72-120 hours
- **Urgent**: 24-48 hours  
- **Stat**: 2-8 hours
- **Distribution**: Gamma distribution for realism

### **Status Distribution**
- **Approved**: ~65%
- **Denied**: ~25%
- **Pended**: ~10%

## 🏥 Health Plan Coverage

The dataset includes realistic health plan names across all 50 states:

**Sample States:**
- **California**: California Health Plan
- **Texas**: Texas Health Solutions
- **Florida**: Florida Health Partners
- **New York**: New York Health Network

**Lines of Business:**
- **Medicaid**: State-sponsored coverage
- **Medicare**: Federal senior coverage
- **Marketplace**: ACA exchange plans

## 📊 Interactive Features

### **Filtering Options**
- **Date Range**: Custom start/end dates
- **State Selection**: Single or multiple states
- **Service Type**: Medical service categories
- **Health Plan**: Specific health plan names
- **Line of Business**: Medicaid/Medicare/Marketplace
- **Status**: Approved/Denied/Pended

### **Real-Time Updates**
- **KPI Cards**: Update instantly with filters
- **Charts**: Redraw with filtered data
- **Data Table**: Sortable, searchable results
- **Export**: Download filtered data as CSV

## 🔧 Technical Details

### **Dependencies**
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
plotly>=5.0.0
python-docx>=0.8.11
```

### **Performance**
- **Memory Efficient**: Batch processing for large datasets
- **Scalable**: Handles 50M+ records
- **Fast Rendering**: Optimized JavaScript for dashboards

### **Data Quality**
- **Realistic Distributions**: Based on industry benchmarks
- **Business Logic**: Healthcare-specific rules and calculations
- **Consistency**: Cross-field validation and relationships

## 📁 File Structure

```
├── build_synthetic_edi278.py          # Main data generator
├── create_working_dashboard.py        # Working dashboard creator
├── create_interactive_dashboard.py    # Power BI-style dashboard
├── create_executive_dashboard.py      # Executive presentation
├── synthetic_edi278_dataset_sample.csv # 100K sample data
├── edi278_working_dashboard.html      # Working dashboard
├── edi278_interactive_dashboard.html  # Interactive dashboard
├── edi278_executive_dashboard.html    # Executive dashboard
├── edi278_*.png                       # Static charts
├── EDI278_Data_Generation_Methodology.docx # Technical documentation
└── README_synthetic_edi278.txt        # Field descriptions
```

## 🎯 Use Cases

### **Technical Demonstrations**
- Showcase data engineering skills
- Demonstrate healthcare domain knowledge
- Present scalable architecture capabilities

### **Executive Presentations**
- Leadership dashboard reviews
- ROI and performance discussions
- Strategic planning sessions

### **Training & Development**
- Healthcare data analysis training
- Dashboard development examples
- Statistical modeling demonstrations

## 🔒 Privacy & Compliance

- **100% Synthetic Data**: No real patient information
- **HIPAA Compliant**: Safe for demonstrations
- **Masked Identifiers**: All PII is synthetic
- **Industry Standard**: Follows healthcare data practices

## 📞 Support

For questions or issues:
1. Check the technical documentation (`EDI278_Data_Generation_Methodology.docx`)
2. Review the field descriptions (`README_synthetic_edi278.txt`)
3. Open an issue in this repository

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install pandas numpy matplotlib seaborn scikit-learn plotly python-docx`
3. **Generate sample data**: `python build_synthetic_edi278.py`
4. **Create dashboards**: `python create_working_dashboard.py`
5. **Open dashboard**: Open `edi278_working_dashboard.html` in your browser

---

**A demo of what I built for EDI at a Fortune 25 health tech company**  
*Enterprise-scale synthetic data with interactive executive dashboards*
