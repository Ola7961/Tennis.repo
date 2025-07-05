# Wimbledon 2025 Gentlemen's Singles Draw Data
# Based on the official draw from wimbledon.com

WIMBLEDON_2025_DRAW = {
    "tournament": "Wimbledon 2025",
    "surface": "Grass",
    "total_players": 128,
    "rounds": ["First Round", "Second Round", "Third Round", "Fourth Round", "Quarter-Finals", "Semi-Finals", "Final"],
    
    # Top seeded players (Top 32 seeds)
    "seeded_players": {
        1: "Jannik Sinner",
        2: "Carlos Alcaraz", 
        3: "Alexander Zverev",
        4: "Jack Draper",
        5: "Taylor Fritz",
        6: "Novak Djokovic",
        7: "Lorenzo Musetti",
        8: "Holger Rune",
        9: "Daniil Medvedev",
        10: "Ben Shelton",
        11: "Alex de Minaur",
        12: "Frances Tiafoe",
        13: "Tommy Paul",
        14: "Andrey Rublev",
        15: "Jakub Mensik",
        16: "Francisco Cerundolo",
        17: "Karen Khachanov",
        18: "Ugo Humbert",
        19: "Grigor Dimitrov",
        20: "Alexei Popyrin",
        21: "Tomas Machac",
        22: "Flavio Cobolli",
        23: "Jiri Lehecka",
        24: "Stefanos Tsitsipas",
        25: "Felix Auger-Aliassime",
        26: "Alejandro Davidovich Fokina",
        27: "Denis Shapovalov",
        28: "Alexander Bublik",
        29: "Brandon Nakashima",
        30: "Alex Michelsen",
        31: "Tallon Griekspoor",
        32: "Matteo Berrettini"
    },
    
    # Quarter sections (each section has 32 players)
    "sections": {
        "Section 1": {
            "top_seed": 1,
            "seed_name": "Jannik Sinner",
            "other_seeds": [32, 16, 17],
            "key_players": [
                "Jannik Sinner", "Matteo Berrettini", "Francisco Cerundolo", "Karen Khachanov",
                "Pedro Martinez", "Grigor Dimitrov", "Sebastian Ofner", "Tommy Paul",
                "Ben Shelton", "Ugo Humbert", "Brandon Nakashima", "Lorenzo Musetti"
            ]
        },
        "Section 2": {
            "top_seed": 4,
            "seed_name": "Jack Draper", 
            "other_seeds": [29, 13, 20],
            "key_players": [
                "Jack Draper", "Brandon Nakashima", "Tommy Paul", "Alexei Popyrin",
                "Marin Cilic", "Flavio Cobolli", "Jakub Mensik", "Alex de Minaur",
                "Tomas Machac", "Miomir Kecmanovic", "Novak Djokovic"
            ]
        },
        "Section 3": {
            "top_seed": 5,
            "seed_name": "Taylor Fritz",
            "other_seeds": [28, 12, 21],
            "key_players": [
                "Taylor Fritz", "Alexander Bublik", "Frances Tiafoe", "Tomas Machac",
                "Alejandro Davidovich Fokina", "Daniil Medvedev", "Nuno Borges", 
                "Karen Khachanov", "Alexander Zverev", "Holger Rune"
            ]
        },
        "Section 4": {
            "top_seed": 2,
            "seed_name": "Carlos Alcaraz",
            "other_seeds": [31, 15, 18],
            "key_players": [
                "Carlos Alcaraz", "Tallon Griekspoor", "Jakub Mensik", "Ugo Humbert",
                "Jiri Lehecka", "Stefanos Tsitsipas", "Felix Auger-Aliassime",
                "Andrey Rublev", "Jan-Lennard Struff", "Fabio Fognini"
            ]
        }
    },
    
    # Potential quarter-final matchups
    "projected_quarterfinals": [
        {
            "match": "QF1",
            "section1": "Section 1",
            "section2": "Section 2", 
            "likely_players": ["Jannik Sinner", "Jack Draper"],
            "seeds": [1, 4]
        },
        {
            "match": "QF2", 
            "section1": "Section 3",
            "section2": "Section 4",
            "likely_players": ["Taylor Fritz", "Carlos Alcaraz"],
            "seeds": [5, 2]
        }
    ],
    
    # Current tournament status (as of July 4, 2025)
    "current_round": "Third Round",
    "remaining_players": 16,
    
    # Notable first round results and upsets
    "notable_results": [
        {
            "round": "First Round",
            "upset": True,
            "winner": "Alexander Zverev",
            "loser": "Cameron Norrie", 
            "seed_upset": False
        },
        {
            "round": "Second Round", 
            "upset": False,
            "winner": "Jannik Sinner",
            "loser": "Aleksandar Vukic",
            "seed_upset": False
        },
        {
            "round": "Second Round",
            "upset": False, 
            "winner": "Carlos Alcaraz",
            "loser": "Fabio Fognini",
            "seed_upset": False
        }
    ],
    
    # Players who have advanced to Third Round (current status)
    "third_round_players": [
        "Jannik Sinner",
        "Pedro Martinez", 
        "Grigor Dimitrov",
        "Sebastian Ofner",
        "Ben Shelton",
        "Brandon Nakashima",
        "Lorenzo Musetti",
        "Flavio Cobolli",
        "Alex de Minaur",
        "Miomir Kecmanovic",
        "Taylor Fritz",
        "Alejandro Davidovich Fokina",
        "Nuno Borges",
        "Karen Khachanov",
        "Andrey Rublev",
        "Carlos Alcaraz"
    ]
}

# Function to get section for a player
def get_player_section(player_name):
    """Get which section a player belongs to"""
    for section_name, section_data in WIMBLEDON_2025_DRAW["sections"].items():
        if player_name in section_data["key_players"]:
            return section_name
    return None

# Function to get potential opponents in next round
def get_potential_opponents(player_name, current_round):
    """Get potential opponents for a player in the next round"""
    section = get_player_section(player_name)
    if not section:
        return []
    
    # This is a simplified version - in reality you'd need to track the bracket more precisely
    section_data = WIMBLEDON_2025_DRAW["sections"][section]
    return [p for p in section_data["key_players"] if p != player_name]

# Function to calculate draw difficulty
def calculate_draw_difficulty(player_name):
    """Calculate relative difficulty of a player's draw section"""
    section = get_player_section(player_name)
    if not section:
        return 0.5
    
    section_data = WIMBLEDON_2025_DRAW["sections"][section]
    
    # Count number of seeded players in section
    seeded_count = len([s for s in section_data["other_seeds"]] + [section_data["top_seed"]])
    
    # Normalize difficulty (more seeds = harder draw)
    difficulty = min(seeded_count / 4.0, 1.0)  # Max 4 seeds per section typically
    
    return difficulty

if __name__ == "__main__":
    # Test the functions
    print("Wimbledon 2025 Draw Information:")
    print(f"Tournament: {WIMBLEDON_2025_DRAW['tournament']}")
    print(f"Current Round: {WIMBLEDON_2025_DRAW['current_round']}")
    print(f"Remaining Players: {WIMBLEDON_2025_DRAW['remaining_players']}")
    
    print("\nTop Seeds:")
    for seed, player in list(WIMBLEDON_2025_DRAW["seeded_players"].items())[:8]:
        section = get_player_section(player)
        difficulty = calculate_draw_difficulty(player)
        print(f"  {seed}. {player} - Section: {section} - Draw Difficulty: {difficulty:.2f}")

