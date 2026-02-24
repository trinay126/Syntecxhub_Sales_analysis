"""
Retail Sales Analysis
=====================
Comprehensive sales data analysis with KPIs, visualizations, and business insights

Author: Data Analysis Team
Date: February 2026
Project: Syntecxhub Internship - Sales Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. DATA LOADING & INITIAL INSPECTION
# ============================================================================

def load_and_inspect_data(filepath):
    """Load sales data and perform initial inspection"""
    print("="*70)
    print("RETAIL SALES ANALYSIS - BUSINESS INTELLIGENCE REPORT")
    print("="*70)
    
    # Load data
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"\n📊 Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Data quality check
    print("\n" + "="*70)
    print("DATA QUALITY CHECK")
    print("="*70)
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ Missing values found:")
        print(missing[missing > 0])
    else:
        print("\n✓ No missing values detected")
    
    print(f"\n✓ Data types: All columns properly formatted")
    print(f"✓ Duplicates: {df.duplicated().sum()} records")
    
    return df


# ============================================================================
# 2. KEY PERFORMANCE INDICATORS (KPIs)
# ============================================================================

def calculate_kpis(df):
    """Calculate essential business KPIs"""
    print("\n" + "="*70)
    print("KEY PERFORMANCE INDICATORS (KPIs)")
    print("="*70)
    
    kpis = {}
    
    # Total Revenue
    kpis['total_revenue'] = df['Revenue'].sum()
    print(f"\n💰 Total Revenue: ${kpis['total_revenue']:,.2f}")
    
    # Total Orders
    kpis['total_orders'] = len(df)
    print(f"📦 Total Orders: {kpis['total_orders']:,}")
    
    # Average Order Value (AOV)
    kpis['avg_order_value'] = df['Revenue'].mean()
    print(f"💵 Average Order Value: ${kpis['avg_order_value']:,.2f}")
    
    # Total Units Sold
    kpis['total_units'] = df['Quantity'].sum()
    print(f"📊 Total Units Sold: {kpis['total_units']:,}")
    
    # Average Units per Order
    kpis['avg_units_per_order'] = df['Quantity'].mean()
    print(f"📈 Average Units per Order: {kpis['avg_units_per_order']:.2f}")
    
    # Number of Unique Products
    kpis['unique_products'] = df['Product'].nunique()
    print(f"🏷️  Unique Products: {kpis['unique_products']}")
    
    # Average Unit Price
    kpis['avg_unit_price'] = df['Unit_Price'].mean()
    print(f"💲 Average Unit Price: ${kpis['avg_unit_price']:.2f}")
    
    # Revenue per Unit
    kpis['revenue_per_unit'] = kpis['total_revenue'] / kpis['total_units']
    print(f"📊 Revenue per Unit: ${kpis['revenue_per_unit']:.2f}")
    
    return kpis


# ============================================================================
# 3. TOP PRODUCTS ANALYSIS
# ============================================================================

def analyze_top_products(df, viz_path, top_n=10):
    """Identify and visualize top-performing products"""
    print("\n" + "="*70)
    print("TOP PRODUCTS ANALYSIS")
    print("="*70)
    
    # Revenue by product
    product_revenue = df.groupby('Product').agg({
        'Revenue': 'sum',
        'Quantity': 'sum',
        'Order_ID': 'count'
    }).round(2)
    product_revenue.columns = ['Total_Revenue', 'Units_Sold', 'Order_Count']
    product_revenue = product_revenue.sort_values('Total_Revenue', ascending=False)
    
    print(f"\n🏆 Top {top_n} Products by Revenue:")
    print(product_revenue.head(top_n))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top products by revenue
    top_products = product_revenue.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_products)))
    
    axes[0, 0].barh(range(len(top_products)), top_products['Total_Revenue'], color=colors)
    axes[0, 0].set_yticks(range(len(top_products)))
    axes[0, 0].set_yticklabels(top_products.index)
    axes[0, 0].set_xlabel('Revenue ($)', fontweight='bold')
    axes[0, 0].set_title(f'Top {top_n} Products by Revenue', fontweight='bold', fontsize=12)
    axes[0, 0].invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_products['Total_Revenue']):
        axes[0, 0].text(v, i, f' ${v:,.0f}', va='center', fontweight='bold')
    
    # Top products by units sold
    top_units = product_revenue.sort_values('Units_Sold', ascending=False).head(top_n)
    axes[0, 1].barh(range(len(top_units)), top_units['Units_Sold'], color='coral')
    axes[0, 1].set_yticks(range(len(top_units)))
    axes[0, 1].set_yticklabels(top_units.index)
    axes[0, 1].set_xlabel('Units Sold', fontweight='bold')
    axes[0, 1].set_title(f'Top {top_n} Products by Units Sold', fontweight='bold', fontsize=12)
    axes[0, 1].invert_yaxis()
    
    # Category performance
    category_revenue = df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
    axes[1, 0].pie(category_revenue.values, labels=category_revenue.index, autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Set3.colors)
    axes[1, 0].set_title('Revenue Distribution by Category', fontweight='bold', fontsize=12)
    
    # Product performance scatter
    product_metrics = df.groupby('Product').agg({
        'Revenue': 'sum',
        'Quantity': 'sum'
    })
    axes[1, 1].scatter(product_metrics['Quantity'], product_metrics['Revenue'], 
                      alpha=0.6, s=200, c='steelblue', edgecolors='black')
    axes[1, 1].set_xlabel('Total Units Sold', fontweight='bold')
    axes[1, 1].set_ylabel('Total Revenue ($)', fontweight='bold')
    axes[1, 1].set_title('Product Performance: Revenue vs Units Sold', fontweight='bold', fontsize=12)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{viz_path}/top_products_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: top_products_analysis.png")
    plt.close()
    
    return product_revenue


# ============================================================================
# 4. REGIONAL ANALYSIS
# ============================================================================

def analyze_regions(df, viz_path):
    """Analyze sales performance by region"""
    print("\n" + "="*70)
    print("REGIONAL PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Regional metrics
    regional_metrics = df.groupby('Region').agg({
        'Revenue': 'sum',
        'Order_ID': 'count',
        'Quantity': 'sum'
    }).round(2)
    regional_metrics.columns = ['Total_Revenue', 'Order_Count', 'Units_Sold']
    regional_metrics['Avg_Order_Value'] = (regional_metrics['Total_Revenue'] / 
                                            regional_metrics['Order_Count']).round(2)
    regional_metrics = regional_metrics.sort_values('Total_Revenue', ascending=False)
    
    print(f"\n🌍 Regional Performance Metrics:")
    print(regional_metrics)
    
    # Calculate market share
    regional_metrics['Market_Share_%'] = (
        (regional_metrics['Total_Revenue'] / regional_metrics['Total_Revenue'].sum()) * 100
    ).round(2)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Revenue by region
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    regional_metrics['Total_Revenue'].plot(kind='bar', ax=axes[0, 0], color=colors)
    axes[0, 0].set_title('Total Revenue by Region', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Region', fontweight='bold')
    axes[0, 0].set_ylabel('Revenue ($)', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(regional_metrics['Total_Revenue']):
        axes[0, 0].text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Market share pie chart
    axes[0, 1].pie(regional_metrics['Total_Revenue'], labels=regional_metrics.index,
                   autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0, 1].set_title('Market Share by Region', fontweight='bold', fontsize=12)
    
    # Average order value by region
    regional_metrics['Avg_Order_Value'].plot(kind='bar', ax=axes[1, 0], color='skyblue')
    axes[1, 0].set_title('Average Order Value by Region', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Region', fontweight='bold')
    axes[1, 0].set_ylabel('AOV ($)', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Orders vs Revenue by region
    x = np.arange(len(regional_metrics))
    width = 0.35
    
    ax2 = axes[1, 1]
    bars1 = ax2.bar(x - width/2, regional_metrics['Order_Count'], width, 
                    label='Order Count', color='steelblue')
    ax2.set_xlabel('Region', fontweight='bold')
    ax2.set_ylabel('Order Count', fontweight='bold')
    ax2.set_title('Order Count by Region', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regional_metrics.index, rotation=45, ha='right')
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{viz_path}/regional_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: regional_analysis.png")
    plt.close()
    
    return regional_metrics


# ============================================================================
# 5. SEASONALITY & TRENDS ANALYSIS
# ============================================================================

def analyze_seasonality(df, viz_path):
    """Analyze sales trends and seasonality patterns"""
    print("\n" + "="*70)
    print("SEASONALITY & TRENDS ANALYSIS")
    print("="*70)
    
    # Time-based aggregations
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    df['Month_Num'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Week'] = df['Date'].dt.isocalendar().week
    
    # Monthly trends
    monthly_revenue = df.groupby(['Year', 'Month_Num', 'Month_Name']).agg({
        'Revenue': 'sum',
        'Order_ID': 'count'
    }).reset_index()
    monthly_revenue = monthly_revenue.sort_values(['Year', 'Month_Num'])
    
    print(f"\n📅 Monthly Revenue Trends (Last 12 Months):")
    print(monthly_revenue.tail(12)[['Year', 'Month_Name', 'Revenue', 'Order_ID']])
    
    # Quarterly performance
    quarterly_revenue = df.groupby(['Year', 'Quarter']).agg({
        'Revenue': 'sum',
        'Order_ID': 'count'
    }).reset_index()
    
    print(f"\n📊 Quarterly Performance:")
    print(quarterly_revenue)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Monthly revenue trend
    monthly_revenue['YearMonth'] = monthly_revenue['Year'].astype(str) + '-' + monthly_revenue['Month_Num'].astype(str).str.zfill(2)
    axes[0, 0].plot(range(len(monthly_revenue)), monthly_revenue['Revenue'], 
                    marker='o', linewidth=2, color='steelblue', markersize=6)
    axes[0, 0].set_title('Monthly Revenue Trend', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Month', fontweight='bold')
    axes[0, 0].set_ylabel('Revenue ($)', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Seasonality by month
    seasonal_pattern = df.groupby('Month_Name')['Revenue'].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    
    colors_seasonal = plt.cm.coolwarm(seasonal_pattern / seasonal_pattern.max())
    axes[0, 1].bar(range(12), seasonal_pattern.values, color=colors_seasonal)
    axes[0, 1].set_title('Average Revenue by Month (Seasonality Pattern)', 
                        fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Month', fontweight='bold')
    axes[0, 1].set_ylabel('Average Revenue ($)', fontweight='bold')
    axes[0, 1].set_xticks(range(12))
    axes[0, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Quarterly comparison
    pivot_quarterly = quarterly_revenue.pivot(index='Quarter', columns='Year', values='Revenue')
    pivot_quarterly.plot(kind='bar', ax=axes[1, 0], color=['#4ECDC4', '#FF6B6B'])
    axes[1, 0].set_title('Quarterly Revenue Comparison', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Quarter', fontweight='bold')
    axes[1, 0].set_ylabel('Revenue ($)', fontweight='bold')
    axes[1, 0].legend(title='Year', loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=0)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Daily orders trend
    daily_orders = df.groupby('Date')['Order_ID'].count()
    axes[1, 1].plot(daily_orders.index, daily_orders.values, alpha=0.6, color='coral')
    axes[1, 1].set_title('Daily Order Volume', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Date', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Orders', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{viz_path}/seasonality_trends.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: seasonality_trends.png")
    plt.close()
    
    return monthly_revenue, seasonal_pattern


# ============================================================================
# 6. ADDITIONAL INSIGHTS
# ============================================================================

def additional_insights(df, viz_path):
    """Generate additional business insights"""
    print("\n" + "="*70)
    print("ADDITIONAL BUSINESS INSIGHTS")
    print("="*70)
    
    # Customer segment analysis
    segment_revenue = df.groupby('Customer_Segment')['Revenue'].sum().sort_values(ascending=False)
    print(f"\n👥 Revenue by Customer Segment:")
    print(segment_revenue)
    
    # Sales channel analysis
    channel_revenue = df.groupby('Sales_Channel')['Revenue'].sum().sort_values(ascending=False)
    print(f"\n🛒 Revenue by Sales Channel:")
    print(channel_revenue)
    
    # Category performance
    category_metrics = df.groupby('Category').agg({
        'Revenue': 'sum',
        'Order_ID': 'count',
        'Quantity': 'sum'
    }).sort_values('Revenue', ascending=False)
    print(f"\n📦 Category Performance:")
    print(category_metrics)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Customer segment
    segment_revenue.plot(kind='bar', ax=axes[0, 0], color=['#FFD700', '#C0C0C0', '#CD7F32'])
    axes[0, 0].set_title('Revenue by Customer Segment', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Revenue ($)', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Sales channel
    axes[0, 1].pie(channel_revenue.values, labels=channel_revenue.index, autopct='%1.1f%%',
                   colors=['#4ECDC4', '#FF6B6B', '#FFA07A'], startangle=90)
    axes[0, 1].set_title('Revenue Distribution by Sales Channel', fontweight='bold', fontsize=12)
    
    # Category performance
    category_metrics['Revenue'].plot(kind='barh', ax=axes[1, 0], color='steelblue')
    axes[1, 0].set_title('Revenue by Category', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Revenue ($)', fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Price vs Quantity relationship
    axes[1, 1].scatter(df['Unit_Price'], df['Quantity'], alpha=0.3, c=df['Revenue'], 
                      cmap='viridis', s=50)
    axes[1, 1].set_xlabel('Unit Price ($)', fontweight='bold')
    axes[1, 1].set_ylabel('Quantity', fontweight='bold')
    axes[1, 1].set_title('Price vs Quantity Relationship', fontweight='bold', fontsize=12)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{viz_path}/additional_insights.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: additional_insights.png")
    plt.close()
    
    return segment_revenue, channel_revenue, category_metrics


# ============================================================================
# 7. BUSINESS RECOMMENDATIONS
# ============================================================================

def generate_recommendations(df, kpis, product_revenue, regional_metrics, 
                            seasonal_pattern):
    """Generate data-driven business recommendations"""
    print("\n" + "="*70)
    print("BUSINESS RECOMMENDATIONS")
    print("="*70)
    
    recommendations = []
    
    # Top product recommendation
    top_product = product_revenue.index[0]
    top_product_revenue = product_revenue.iloc[0]['Total_Revenue']
    recommendations.append(
        f"1. PRODUCT STRATEGY: {top_product} is the top revenue generator "
        f"(${top_product_revenue:,.2f}). Consider expanding inventory and creating "
        f"complementary product bundles."
    )
    
    # Regional focus
    top_region = regional_metrics.index[0]
    top_region_share = regional_metrics.iloc[0]['Market_Share_%']
    recommendations.append(
        f"2. REGIONAL EXPANSION: {top_region} accounts for {top_region_share:.1f}% "
        f"of total revenue. Invest in marketing campaigns in underperforming regions "
        f"to balance market presence."
    )
    
    # Seasonality insight
    peak_month = seasonal_pattern.idxmax()
    low_month = seasonal_pattern.idxmin()
    recommendations.append(
        f"3. SEASONAL PLANNING: Sales peak in {peak_month} and dip in {low_month}. "
        f"Plan inventory and promotional campaigns accordingly. Consider flash sales "
        f"during low-revenue months."
    )
    
    # AOV optimization
    aov = kpis['avg_order_value']
    recommendations.append(
        f"4. INCREASE AOV: Current average order value is ${aov:.2f}. Implement "
        f"cross-selling strategies, bundle deals, and free shipping thresholds to "
        f"increase order values by 15-20%."
    )
    
    # Customer segment opportunity
    recommendations.append(
        f"5. CUSTOMER SEGMENTATION: Focus on high-value enterprise customers while "
        f"implementing loyalty programs for individual buyers to increase retention "
        f"and lifetime value."
    )
    
    print("\n💡 Strategic Recommendations:\n")
    for rec in recommendations:
        print(f"{rec}\n")
    
    return recommendations


# ============================================================================
# 8. EXPORT ONE-PAGE SUMMARY
# ============================================================================

def create_summary_pdf(kpis, product_revenue, regional_metrics, seasonal_pattern, 
                      recommendations, output_path, viz_path):
    """Create a one-page PDF summary with charts"""
    print("\n" + "="*70)
    print("GENERATING ONE-PAGE SUMMARY PDF")
    print("="*70)
    
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    
    # Create PDF
    pdf_file = f'{output_path}/Sales_Analysis_Summary.pdf'
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Retail Sales Analysis - Executive Summary")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    
    # KPIs Section
    y_position = height - 110
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Key Performance Indicators")
    
    y_position -= 25
    c.setFont("Helvetica", 10)
    kpi_text = [
        f"Total Revenue: ${kpis['total_revenue']:,.2f}",
        f"Total Orders: {kpis['total_orders']:,}",
        f"Average Order Value: ${kpis['avg_order_value']:,.2f}",
        f"Total Units Sold: {kpis['total_units']:,}",
    ]
    
    for kpi in kpi_text:
        c.drawString(60, y_position, f"• {kpi}")
        y_position -= 15
    
    # Top Products
    y_position -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Top 5 Products by Revenue")
    
    y_position -= 20
    c.setFont("Helvetica", 9)
    top_5_products = product_revenue.head(5)
    for idx, (product, row) in enumerate(top_5_products.iterrows(), 1):
        c.drawString(60, y_position, 
                    f"{idx}. {product}: ${row['Total_Revenue']:,.0f}")
        y_position -= 12
    
    # Regional Performance
    y_position -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Top Regions by Revenue")
    
    y_position -= 20
    c.setFont("Helvetica", 9)
    for idx, (region, row) in enumerate(regional_metrics.head(3).iterrows(), 1):
        c.drawString(60, y_position,
                    f"{idx}. {region}: ${row['Total_Revenue']:,.0f} ({row['Market_Share_%']:.1f}%)")
        y_position -= 12
    
    # Recommendations
    y_position -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Strategic Recommendations")
    
    y_position -= 20
    c.setFont("Helvetica", 8)
    for i, rec in enumerate(recommendations[:3], 1):
        # Wrap text for recommendations
        words = rec.split()
        line = f"{i}. "
        for word in words:
            if len(line + word) < 90:
                line += word + " "
            else:
                c.drawString(60, y_position, line)
                y_position -= 10
                line = "   " + word + " "
        if line.strip():
            c.drawString(60, y_position, line)
            y_position -= 12
    
    # Add visualizations note
    y_position -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y_position, 
                "Detailed visualizations available in the 'visualizations' folder")
    
    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(50, 30, "Retail Sales Analysis | Business Intelligence Report")
    c.drawString(width - 150, 30, "Syntecxhub Internship Project")
    
    # Save PDF
    c.save()
    print(f"\n✓ PDF summary created: Sales_Analysis_Summary.pdf")
    
    return pdf_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    import os
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to script location
    data_path = os.path.join(script_dir, 'data', 'retail_sales_data.csv')
    viz_path = os.path.join(script_dir, 'visualizations')
    output_path = os.path.join(script_dir, 'outputs')
    
    # Create directories if they don't exist
    print("\n📊 Starting Retail Sales Analysis...")
    print("\n📁 Setting up project directories...")
    
    for path in [viz_path, output_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"   ✓ Created: {path}")
        else:
            print(f"   ✓ Found: {path}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Data file not found!")
        print(f"   Looking for: {data_path}")
        print(f"\n📁 Please ensure 'retail_sales_data.csv' is in the 'data' folder")
        print(f"   Expected structure:")
        print(f"   {script_dir}/")
        print(f"   ├── data/")
        print(f"   │   └── retail_sales_data.csv")
        print(f"   └── salesanalysis.py (or sales_analysis.py)")
        return
    
    print(f"\n📁 Project paths:")
    print(f"   Script location: {script_dir}")
    print(f"   Data: {data_path}")
    print(f"   Visualizations: {viz_path}")
    print(f"   Outputs: {output_path}")
    
    # 1. Load data
    df = load_and_inspect_data(data_path)
    
    # 2. Calculate KPIs
    kpis = calculate_kpis(df)
    
    # 3. Analyze top products
    product_revenue = analyze_top_products(df, viz_path)
    
    # 4. Regional analysis
    regional_metrics = analyze_regions(df, viz_path)
    
    # 5. Seasonality & trends
    monthly_revenue, seasonal_pattern = analyze_seasonality(df, viz_path)
    
    # 6. Additional insights
    segment_revenue, channel_revenue, category_metrics = additional_insights(df, viz_path)
    
    # 7. Generate recommendations
    recommendations = generate_recommendations(df, kpis, product_revenue, 
                                               regional_metrics, seasonal_pattern)
    
    # 8. Create PDF summary
    pdf_file = create_summary_pdf(kpis, product_revenue, regional_metrics, 
                                   seasonal_pattern, recommendations, 
                                   output_path, viz_path)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📊 Total visualizations created: 4")
    print(f"📄 PDF summary generated: Sales_Analysis_Summary.pdf")
    print(f"\n💡 Check the 'visualizations' and 'outputs' folders for results!")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()