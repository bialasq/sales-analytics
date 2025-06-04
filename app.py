from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import json
import random
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Przechowywanie danych w pamięci (zamiast bazy danych)
sales_data = []

class SalesAgent:
    """Agent handlowy - analiza sprzedaży i relacji z klientami"""
    
    def analyze_sales_performance(self, df):
        insights = []
        
        # Analiza konwersji
        if 'status' in df.columns or 'zamowienie' in df.columns.str.lower():
            status_col = next((col for col in df.columns if 'status' in col.lower() or 'zamowienie' in col.lower()), None)
            if status_col:
                conversion_rate = (df[status_col].str.contains('zamówione|sprzedane|tak', case=False, na=False).sum() / len(df)) * 100
                insights.append(f"Wskaźnik konwersji wynosi {conversion_rate:.1f}%. " + 
                              ("Bardzo dobry wynik!" if conversion_rate > 70 else 
                               "Wymaga poprawy - skup się na follow-up." if conversion_rate < 40 else
                               "Akceptowalny poziom."))
        
        # Analiza klientów
        if any('klient' in col.lower() for col in df.columns):
            client_col = next(col for col in df.columns if 'klient' in col.lower())
            repeat_clients = df[client_col].value_counts()
            loyal_clients = (repeat_clients > 1).sum()
            insights.append(f"Masz {loyal_clients} stałych klientów z {len(repeat_clients)} wszystkich. " +
                          "Skup się na budowaniu długotrwałych relacji.")
        
        # Analiza czasowa
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():
                    best_day = df.groupby(df[col].dt.day_name()).size().idxmax()
                    insights.append(f"Najlepszy dzień sprzedaży to {best_day}. Planuj ważne spotkania w ten dzień.")
                    break
            except:
                continue
                
        return insights
    
    def recommend_improvements(self, df):
        recommendations = []
        
        # Rekomendacje produktowe
        if any('produkt' in col.lower() for col in df.columns):
            product_col = next(col for col in df.columns if 'produkt' in col.lower())
            top_products = df[product_col].value_counts().head(3)
            recommendations.append(f"Twoje top produkty: {', '.join(top_products.index)}. " +
                                 "Skoncentruj się na cross-sellingu podobnych produktów.")
        
        # Rekomendacje cenowe
        if any('cena' in col.lower() or 'wartosc' in col.lower() for col in df.columns):
            price_col = next((col for col in df.columns if 'cena' in col.lower() or 'wartosc' in col.lower()), None)
            if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
                avg_price = df[price_col].mean()
                recommendations.append(f"Średnia wartość transakcji: {avg_price:.0f} zł. " +
                                     "Próbuj zwiększyć wartość koszyka przez bundling produktów.")
        
        recommendations.extend([
            "Regularnie kontaktuj się z klientami - minimum raz w miesiącu",
            "Stwórz system rekomendacji dla każdego typu klienta",
            "Używaj technik storytelling przy prezentacji produktów"
        ])
        
        return recommendations

class EconomicAgent:
    """Agent ekonomiczny - analiza finansowa i rentowności"""
    
    def analyze_financial_performance(self, df):
        insights = []
        
        # Analiza przychodów
        revenue_cols = [col for col in df.columns if any(term in col.lower() for term in ['przychod', 'wartosc', 'cena', 'kwota'])]
        if revenue_cols:
            revenue_col = revenue_cols[0]
            if pd.api.types.is_numeric_dtype(df[revenue_col]):
                total_revenue = df[revenue_col].sum()
                avg_transaction = df[revenue_col].mean()
                insights.append(f"Całkowity przychód: {total_revenue:,.0f} zł, średnia transakcja: {avg_transaction:.0f} zł")
                
                # Analiza trendu
                if len(df) > 10:
                    first_half = df[:len(df)//2][revenue_col].mean()
                    second_half = df[len(df)//2:][revenue_col].mean()
                    growth = ((second_half - first_half) / first_half) * 100
                    insights.append(f"Trend przychodów: {growth:+.1f}%. " + 
                                  ("Świetna dynamika wzrostu!" if growth > 10 else
                                   "Stabilizacja - szukaj nowych możliwości." if abs(growth) < 5 else
                                   "Spadek wymaga natychmiastowej akcji!"))
        
        # Analiza kosztów (jeśli dostępne)
        cost_cols = [col for col in df.columns if any(term in col.lower() for term in ['koszt', 'wydatek'])]
        if cost_cols and revenue_cols:
            cost_col = cost_cols[0]
            if pd.api.types.is_numeric_dtype(df[cost_col]):
                margin = ((df[revenue_cols[0]].sum() - df[cost_col].sum()) / df[revenue_cols[0]].sum()) * 100
                insights.append(f"Marża zysku: {margin:.1f}%. " + 
                              ("Doskonała rentowność!" if margin > 30 else
                               "Optymalizuj koszty." if margin < 15 else
                               "Dobry poziom marży."))
        
        return insights
    
    def forecast_trends(self, df):
        forecasts = []
        
        # Prognoza sezonowości
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():
                    monthly_sales = df.groupby(df[col].dt.month).size()
                    best_month = monthly_sales.idxmax()
                    worst_month = monthly_sales.idxmin()
                    month_names = {1:'Styczeń', 2:'Luty', 3:'Marzec', 4:'Kwiecień', 5:'Maj', 6:'Czerwiec',
                                 7:'Lipiec', 8:'Sierpień', 9:'Wrzesień', 10:'Październik', 11:'Listopad', 12:'Grudzień'}
                    forecasts.append(f"Sezonowość: najlepszy miesiąc to {month_names.get(best_month, best_month)}, " +
                                   f"najsłabszy: {month_names.get(worst_month, worst_month)}")
                    break
            except:
                continue
        
        forecasts.extend([
            "Przewiduj 15-20% wzrost sprzedaży przy optymalizacji procesów",
            "Inwestuj w digitalizację - ROI około 300% w ciągu roku",
            "Planuj budżet marketingowy na poziomie 8-12% przychodów"
        ])
        
        return forecasts

class RouteAgent:
    """Agent tras sprzedażowych - optymalizacja wizyt handlowych"""
    
    def analyze_customer_priority(self, df):
        priorities = []
        
        # Analiza wartości klientów
        if any('klient' in col.lower() for col in df.columns):
            client_col = next(col for col in df.columns if 'klient' in col.lower())
            
            # Wartość klienta
            value_cols = [col for col in df.columns if any(term in col.lower() for term in ['wartosc', 'cena', 'przychod'])]
            if value_cols:
                value_col = value_cols[0]
                if pd.api.types.is_numeric_dtype(df[value_col]):
                    client_values = df.groupby(client_col)[value_col].agg(['sum', 'count', 'mean']).round(0)
                    client_values['priority_score'] = (
                        client_values['sum'] * 0.5 + 
                        client_values['count'] * 100 * 0.3 + 
                        client_values['mean'] * 0.2
                    )
                    top_clients = client_values.nlargest(5, 'priority_score')
                    
                    priorities.append("🏆 KLIENCI PRIORYTETOWI (wizyta co tydzień):")
                    for client, data in top_clients.iterrows():
                        priorities.append(f"• {client}: {data['sum']:.0f} zł wartości, {data['count']} transakcji")
        
        return priorities
    
    def suggest_route_optimization(self, df):
        routes = []
        
        # Lokalizacja klientów (jeśli dostępna)
        location_cols = [col for col in df.columns if any(term in col.lower() for term in ['miasto', 'region', 'lokalizacja', 'adres'])]
        if location_cols:
            location_col = location_cols[0]
            regions = df[location_col].value_counts()
            
            routes.append("🗺️ OPTYMALIZACJA TRAS:")
            for region, count in regions.head(3).items():
                routes.append(f"• {region}: {count} klientów - zaplanuj dzień wizyt")
        
        # Harmonogram wizyt
        routes.extend([
            "\n📅 DWUTYGODNIOWY PLAN WIZYT:",
            "• TYDZIEŃ 1: Klienci A-priorytet + 2-3 nowych prospektów",
            "• TYDZIEŃ 2: Klienci B-priorytet + follow-up po ofertach",
            "• Poniedziałki: planowanie i cold calling",
            "• Wtorki-Czwartki: wizyty u klientów",
            "• Piątki: raporty i przygotowanie ofert"
        ])
        
        return routes
    
    def recommend_visit_frequency(self, df):
        recommendations = []
        
        if any('klient' in col.lower() for col in df.columns):
            client_col = next(col for col in df.columns if 'klient' in col.lower())
            visit_frequency = df[client_col].value_counts()
            
            recommendations.append("⏰ CZĘSTOTLIWOŚĆ WIZYT:")
            recommendations.append("• Klienci 5+ transakcji: co tydzień")
            recommendations.append("• Klienci 2-4 transakcje: co 2 tygodnie") 
            recommendations.append("• Klienci 1 transakcja: co miesiąc")
            recommendations.append("• Nowi prospekty: follow-up po 3 dniach")
        
        return recommendations

# Instancje agentów
sales_agent = SalesAgent()
economic_agent = EconomicAgent()
route_agent = RouteAgent()

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Brak pliku'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nie wybrano pliku'}), 400
        
        # Czytanie pliku Excel
        df = pd.read_excel(file)
        
        # Podstawowe przetwarzanie danych
        global sales_data
        sales_data = df.to_dict('records')
        
        return jsonify({
            'message': 'Plik przesłany pomyślnie',
            'rows': len(sales_data),
            'columns': list(df.columns)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(sales_data)

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    if not sales_data:
        return jsonify({'error': 'Brak danych'}), 400
    
    try:
        df = pd.DataFrame(sales_data)
        
        # Podstawowe analizy
        analytics = {
            'total_records': len(df),
            'summary_stats': {},
            'monthly_trends': [],
            'top_products': []
        }
        
        # Statystyki dla kolumn numerycznych
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            analytics['summary_stats'][col] = {
                'mean': float(df[col].mean()),
                'sum': float(df[col].sum()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        
        # Trendy miesięczne (jeśli jest kolumna z datą)
        date_columns = df.select_dtypes(include=['datetime64', 'object']).columns
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():
                    monthly = df.groupby(df[col].dt.to_period('M')).size()
                    analytics['monthly_trends'] = [
                        {'month': str(period), 'count': int(count)}
                        for period, count in monthly.items()
                    ]
                    break
            except:
                continue
        
        # Top produkty (jeśli są kolumny tekstowe)
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if 'product' in col.lower() or 'nazwa' in col.lower():
                top_products = df[col].value_counts().head(5)
                analytics['top_products'] = [
                    {'name': str(name), 'count': int(count)}
                    for name, count in top_products.items()
                ]
                break
        
        return jsonify(analytics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart-data', methods=['GET'])
def get_chart_data():
    if not sales_data:
        return jsonify({'error': 'Brak danych'}), 400
    
    try:
        df = pd.DataFrame(sales_data)
        
        # Dane dla wykresów
        chart_data = {
            'line_chart': [],
            'bar_chart': [],
            'pie_chart': []
        }
        
        # Przykładowe dane dla wykresów
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            # Wykres liniowy - trendy
            first_numeric = numeric_columns[0]
            chart_data['line_chart'] = [
                {'x': i, 'y': float(val)}
                for i, val in enumerate(df[first_numeric].head(10))
            ]
            
            # Wykres słupkowy
            if len(numeric_columns) > 1:
                second_numeric = numeric_columns[1]
                chart_data['bar_chart'] = [
                    {'label': f'Item {i}', 'value': float(val)}
                    for i, val in enumerate(df[second_numeric].head(5))
                ]
        
        # Wykres kołowy
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            first_text = text_columns[0]
            pie_data = df[first_text].value_counts().head(5)
            chart_data['pie_chart'] = [
                {'label': str(label), 'value': int(value)}
                for label, value in pie_data.items()
            ]
        
        return jsonify(chart_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-analysis', methods=['GET'])
def get_ai_analysis():
    if not sales_data:
        return jsonify({'error': 'Brak danych do analizy'}), 400
    
    try:
        df = pd.DataFrame(sales_data)
        
        # Analiza przez agentów AI
        analysis = {
            'sales_insights': sales_agent.analyze_sales_performance(df),
            'sales_recommendations': sales_agent.recommend_improvements(df),
            'financial_analysis': economic_agent.analyze_financial_performance(df),
            'economic_forecast': economic_agent.forecast_trends(df),
            'customer_priorities': route_agent.analyze_customer_priority(df),
            'route_optimization': route_agent.suggest_route_optimization(df),
            'visit_schedule': route_agent.recommend_visit_frequency(df),
            'summary': generate_executive_summary(df)
        }
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_executive_summary(df):
    """Generuje podsumowanie wykonawcze na podstawie analiz wszystkich agentów"""
    
    summary = {
        'period_overview': '',
        'key_metrics': [],
        'main_recommendations': [],
        'action_plan': []
    }
    
    # Przegląd okresu
    if len(df) > 0:
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():
                    start_date = df[col].min().strftime('%Y-%m-%d')
                    end_date = df[col].max().strftime('%Y-%m-%d')
                    summary['period_overview'] = f"Analiza obejmuje okres od {start_date} do {end_date} ({len(df)} rekordów)"
                    break
            except:
                continue
        
        if not summary['period_overview']:
            summary['period_overview'] = f"Analiza obejmuje {len(df)} rekordów sprzedażowych"
    
    # Kluczowe metryki
    revenue_cols = [col for col in df.columns if any(term in col.lower() for term in ['przychod', 'wartosc', 'cena', 'kwota'])]
    if revenue_cols and pd.api.types.is_numeric_dtype(df[revenue_cols[0]]):
        total_revenue = df[revenue_cols[0]].sum()
        avg_transaction = df[revenue_cols[0]].mean()
        summary['key_metrics'].extend([
            f"💰 Całkowity przychód: {total_revenue:,.0f} zł",
            f"📊 Średnia transakcja: {avg_transaction:.0f} zł",
            f"📈 Liczba transakcji: {len(df)}"
        ])
    
    if any('klient' in col.lower() for col in df.columns):
        client_col = next(col for col in df.columns if 'klient' in col.lower())
        unique_clients = df[client_col].nunique()
        summary['key_metrics'].append(f"👥 Liczba klientów: {unique_clients}")
    
    # Główne rekomendacje (top 3)
    summary['main_recommendations'] = [
        "🎯 Skoncentruj się na klientach o najwyższej wartości - zapewnią 80% przychodów",
        "📱 Zdigitalizuj proces sprzedaży - CRM + automatyzacja follow-up",
        "🤝 Buduj długotrwałe relacje - koszt pozyskania nowego klienta jest 5x wyższy"
    ]
    
    # Plan działania (2 tygodnie)
    summary['action_plan'] = [
        "TYDZIEŃ 1: Przegląd i kontakt z top 10 klientami",
        "TYDZIEŃ 2: Implementacja nowego systemu wizyt",
        "Codziennie: 2h na cold calling, 6h na wizyty",
        "Co tydzień: analiza wyników i optymalizacja tras"
    ]
    
    return summary

if __name__ == '__main__':
    app.run(debug=True, port=5000)

    from flask_sqlalchemy import SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sales.db'
db = SQLAlchemy(app)