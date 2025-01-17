import nltk
from nltk.corpus import wordnet

# Ensure nltk resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def compare_phrases(phrase1, phrase2):
    words1 = phrase1.split()
    words2 = phrase2.split()
    
    # Compare word pairs
    for word1 in words1:
        for word2 in words2:
            synonyms1 = set(lemma.name().lower() for syn in wordnet.synsets(word1) for lemma in syn.lemmas())
            synonyms2 = set(lemma.name().lower() for syn in wordnet.synsets(word2) for lemma in syn.lemmas())
            
            if synonyms1 & synonyms2:
                return f"The phrases '{phrase1}' and '{phrase2}' share synonyms between '{word1}' and '{word2}'."
    
    return f"The phrases '{phrase1}' and '{phrase2}' do not share synonyms."

# Example Usage
print(compare_phrases("fast river", "rapid stream"))


unique_company_words = [
    "Analytics", "Biotechnology", "Sustainability", "Architectural", "Biosciences", 
    "Carriers", "Cloud", "Communications", "Construction", "Distributors", 
    "Diversified", "Engineering", "Enterprises", "Exporters", "Fabrics", "Finance", 
    "Foods", "Gases", "Healthcare", "Imports", "Innovations", "Insurance", 
    "Logistics", "Management", "Manufacturers", "Marketing", "Mobile", "Networking", 
    "Packaging", "Petroleum", "Pharmaceuticals", "Products", "Renewable", "Resources", 
    "Retailing", "Security", "Securities", "Shipping", "Studios", "Sustainability", 
    "Systems", "Technology", "Textiles", "Telecom", "Transportation", "Trading", 
    "Toys", "Utilities", "Ventures", "Water", "Wholesale", "Wholesale", 
    "Transportation", "Advisory", "Agriculture", "Automation", "Banking", 
    "Branding", "Chemical", "Computers", "Consulting", "Consumer", "Contractors", 
    "Consultants", "Cultural", "Designers", "Distribution", "Digital", "Development", 
    "Energy", "Electronics", "Entertainment", "Exports", "Engineering", "Furnishing", 
    "Global", "Growth", "Health", "Housing", "Innovation", "Investment", 
    "Logistics", "Management", "Mining", "Manufacturing", "Market", "Medical", 
    "Media", "Metal", "Mobile", "Natural", "Organics", "Outfitters", "Pharmaceutical", 
    "Packaging", "Plastics", "Precious", "Professional", "Projects", "Real Estate", 
    "Research", "Renewables", "Retail", "Solutions", "Security", "Scientific", 
    "Software", "Supply", "Trading", "Transport", "Utilities", "Venture", "Wind", 
    "Wealth", "Wireless", "Waste", "Zoological", "Environmental", "Electrical", 
    "Engineering", "Advanced", "Agency", "Academy", "Apparel", "Automotive", 
    "Financial", "Hardware", "Healthcare", "Infrastructure", "Insights", 
    "Laboratories", "Manufacturing", "Personal", "Recyclers", "Security", "Suppliers", 
    "Textile", "Telecommunications", "Visual", "Welding", "Wood", "Creative", 
    "Capital", "Consulting", "Construction", "Healthcare", "Software", "Systems"
]
company_words_300 = [
    "Inc.", "Ltd.", "Corporation", "Corp.", "LLC", "Group", "Co.", "Company", 
    "Enterprises", "Technologies", "Solutions", "Partners", "Systems", "International", 
    "Global", "Holdings", "Industries", "Associates", "Consulting", "Services", 
    "Ventures", "PLC", "Firm", "Development", "Marketing", "Corporation", "S.A.", 
    "GmbH", "B.V.", "Pvt", "LLP", "SA", "Sdn Bhd", "Cooperative", "Trust", 
    "Agency", "Union", "Fund", "Bank", "Institute", "Research", "Laboratories", 
    "Networks", "Products", "Solutions", "Realty", "Properties", "Construction", 
    "Tech", "Software", "Design", "Studio", "Group", "Works", "Industries", "Corporation", 
    "Limited", "LLP", "Capital", "Market", "Management", "Associates", "Limited Liability", 
    "Enterprises", "Energy", "Network", "Marketing", "Manufacturing", "Brand", "Advisors", 
    "Distribution", "Importers", "Exporters", "Consultants", "Financial", "Retail", 
    "Securities", "Manufacturers", "Construction", "Logistics", "Materials", "Services", 
    "Retail", "Agency", "Brokerage", "Healthcare", "Insurance", "Resort", "Apparel", 
    "Technology", "Advanced", "Electric", "Energy", "Communications", "Research", "Solutions", 
    "Design", "Software", "Enterprises", "Medical", "Labs", "Pharmaceutical", "Real Estate", 
    "Distribution", "Group", "Ventures", "Corporation", "Limited Liability", "Investments", 
    "Network", "Development", "Trade", "Resources", "Associates", "International", "Partners", 
    "Advanced", "Strategic", "Health", "Group", "Consulting", "Realty", "Financial", 
    "Manufacturers", "Technologies", "Corporation", "Studios", "Technologies", "Research", 
    "International", "Chemical", "Digital", "Cloud", "Capital", "Engineering", 
    "Importers", "Electronics", "Global", "Energy", "Transport", "Architects", 
    "Insurance", "Designs", "Projects", "Products", "Trust", "Gases", "Solutions", 
    "Producers", "Industries", "Worldwide", "Agency", "Communications", "System", 
    "Entertainment", "Healthcare", "Banking", "Manufacturing", "Realtors", "Builders", 
    "Fabrics", "Land", "Mining", "Logistics", "Packaging", "Technology", "Research", 
    "Petroleum", "Biosciences", "Energy", "Consulting", "Retail", "Investments", "Real Estate", 
    "Recycling", "Manufacturers", "Suppliers", "Corporations", "Industries", "Electric", 
    "Properties", "Branding", "Designers", "Packaging", "Urban", "Water", "Network", 
    "Logistics", "Manufacturing", "Trading", "Renewable", "Realty", "Brokers", "Pharmaceutical", 
    "Digital", "Mobile", "Security", "Electronics", "Retailers", "Telecom", "Retailers", 
    "Transport", "Imports", "Exports", "Mining", "Studios", "Biotech", "Consulting", 
    "Gases", "Products", "Studios", "Manufacturing", "Development", "Consulting", 
    "Entertainment", "Foods", "Tech", "Apparel", "Sustainability", "Builders", "Carriers", 
    "Communications", "Biotechnology", "Producers", "Distributors", "Property", "Organics", 
    "Textiles", "Real Estate", "Designers", "Construction", "Shipping", "Architectural", 
    "Design", "Developers", "Retailing", "Laboratories", "Packaging", "Analytics", 
    "Trading", "Solutions", "Warehousing", "Consultants", "Recyclers", "Clothing", 
    "Transportation", "Insurance", "Financials", "Corporations", "Diversified", 
    "Contractors", "Energy", "Brokers", "Environmental", "Management", "Builders", 
    "Advisory", "Trading", "Toys", "Automotive", "Logistics", "Corporation", 
    "Investment", "Consumer", "Finance", "Project", "Retailers", "Electronic", 
    "Marketing", "Health", "Realty", "Solutions", "Security", "Trade", "Transports", 
    "Distribution", "Outfitters", "Consultants", "Pharmaceuticals", "Computers", "Industries",
    "Exporters", "Investment Group", "Shipping", "Broker", "Consulting Group", "Trading Co.", 
    "Packaging Co.", "Manufacturers Group", "Electronics Inc.", "Software Corp.", "Retail Ltd.",
    "Apparel Co.", "System Co.", "BioTech", "Renewable Energy", "Investment Partners", 
    "Recyclers Inc.", "Medical Group", "Financial Advisors", "Logistics Inc.", "Consulting Firm",
    "Tech Solutions", "Global Ventures", "Research Group", "Manufacturing Co.", "Electronics Group", 
    "Telecom Solutions", "Chemical Co.", "Biotech Solutions", "Financial Management", 
    "Healthcare Solutions", "Engineering Services", "Sustainability Group", "Retail Partners", 
    "Renewable Solutions", "Trading Ltd.", "Telecom Group", "Electronics Solutions", 
    "Media Group", "Exporters Group", "Financial Group", "Technology Inc.", "Consumer Goods", 
    "Construction Ltd.", "Energy Group", "Design Co.", "Security Inc.", "Real Estate Partners", 
    "Technology Partners", "Pharmaceutical Solutions", "Mining Ltd.", "Consulting Partners", 
    "Energy Solutions", "Logistics Group", "Apparel Group", "Engineering Group", "Real Estate Ltd."
]


common_company_words_extended_2 = [
    "Inc.", "Ltd.", "Corporation", "Corp.", "LLC", "Group", "Co.", "Company", 
    "Enterprises", "Technologies", "Solutions", "Partners", "Systems", "International", 
    "Global", "Holdings", "Industries", "Associates", "Consulting", "Services", 
    "Ventures", "PLC", "Firm", "Development", "Marketing", "Corporation", "S.A.", 
    "GmbH", "B.V.", "Pvt", "LLP", "SA", "Sdn Bhd", "Cooperative", "Trust", 
    "Agency", "Union", "Fund", "Bank", "Institute", "Research", "Laboratories", 
    "Networks", "Products", "Solutions", "Realty", "Properties", "Construction", 
    "Tech", "Software", "Design", "Studio", "Group", "Works", "Industries", "Corporation", 
    "Limited", "LLP", "Capital", "Market", "Management", "Associates", "Limited Liability", 
    "Enterprises", "Energy", "Network", "Marketing", "Manufacturing", "Brand", "Advisors", 
    "Distribution", "Importers", "Exporters", "Consultants", "Financial", "Retail", 
    "Securities", "Manufacturers", "Construction", "Logistics", "Materials", "Services", 
    "Retail", "Agency", "Brokerage", "Healthcare", "Insurance", "Resort", "Apparel", 
    "Technology", "Advanced", "Electric", "Energy", "Communications", "Research", "Solutions", 
    "Design", "Software", "Enterprises", "Medical", "Labs", "Pharmaceutical", "Real Estate", 
    "Distribution", "Group", "Ventures", "Corporation", "Limited Liability", "Investments", 
    "Network", "Development", "Trade", "Resources", "Associates", "International", "Partners", 
    "Advanced", "Strategic", "Health", "Group", "Consulting", "Realty", "Financial", 
    "Manufacturers", "Technologies", "Corporation", "Studios", "Technologies", "Research", 
    "International", "Chemical", "Digital", "Cloud", "Capital", "Engineering", 
    "Importers", "Electronics", "Global", "Energy", "Transport", "Architects", 
    "Insurance", "Designs", "Projects", "Products", "Trust", "Gases", "Solutions", 
    "Producers", "Industries", "Worldwide", "Agency", "Communications", "System", 
    "Entertainment", "Healthcare", "Banking", "Manufacturing", "Realtors", "Builders", 
    "Fabrics", "Land", "Mining", "Logistics", "Packaging", "Technology", "Research", 
    "Petroleum", "Biosciences", "Energy", "Consulting", "Retail", "Investments", "Real Estate", 
    "Recycling", "Manufacturers", "Suppliers", "Corporations", "Industries", "Electric", 
    "Properties", "Branding", "Designers", "Packaging", "Urban", "Water", "Network", 
    "Logistics", "Manufacturing", "Trading", "Renewable", "Realty", "Brokers", "Pharmaceutical", 
    "Digital", "Mobile", "Security", "Electronics", "Retailers", "Telecom", "Retailers", 
    "Transport", "Imports", "Exports", "Mining", "Studios", "Biotech", "Consulting", 
    "Gases", "Products", "Studios", "Manufacturing", "Development", "Consulting", 
    "Entertainment", "Foods", "Tech", "Apparel", "Sustainability", "Builders", "Carriers", 
    "Communications", "Biotechnology", "Producers", "Distributors", "Property", "Organics", 
    "Textiles", "Real Estate", "Designers", "Construction", "Shipping", "Architectural", 
    "Design", "Developers", "Retailing", "Laboratories", "Packaging", "Analytics", 
    "Trading", "Solutions", "Warehousing", "Consultants", "Recyclers", "Clothing", 
    "Transportation", "Insurance", "Financials", "Corporations", "Diversified", 
    "Contractors", "Energy", "Brokers", "Environmental", "Management", "Builders", 
    "Advisory", "Trading", "Toys", "Automotive", "Logistics", "Corporation", 
    "Investment", "Consumer", "Finance", "Project", "Retailers", "Electronic", 
    "Marketing", "Health", "Realty", "Solutions", "Security", "Trade", "Transports", 
    "Distribution", "Outfitters", "Consultants", "Pharmaceuticals", "Computers", "Industries"
]

