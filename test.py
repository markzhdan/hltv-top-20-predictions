import re


def extract_and_clean_in_game_names(text):
    # Regular expression to find names within quotes
    pattern = r'"([^"]+)"'
    # Find all matches of the pattern
    matches = re.findall(pattern, text)

    # Clean each name by stripping and removing unwanted characters if necessary
    cleaned_names = {match.strip() for match in matches}

    # Convert the set back to a list to maintain list type for the return value
    # Using a set also ensures duplicates are removed
    unique_cleaned_names = list(cleaned_names)
    return unique_cleaned_names


# Example usage with a portion of the input
text_segment = """
1. France Mathieu "ZywOo" Herbaut
2. Bosnia and Herzegovina Nikola "⁠NiKo⁠" Kovač
3. Estonia Robin "ropz" Kool
4. Russia Ilya "m0NESY" Osipov
5. Israel Lotan "Spinx" Giladi
6. Spain Álvaro "SunPayus" García
7. Ukraine Oleksandr "⁠s1mple"⁠ Kostyliev
8. Russia Dmitry "sh1ro" Sokolov
9. Denmark Martin "stavn" Lund
10. Latvia Helvijs "⁠broky⁠" Saukants
11. Denmark Nicolai "device" Reedtz
12. Slovakia David "frozen" Čerňanský
13. Bosnia and Herzegovina Nemanja "huNter-⁠" Kovač
14. Israel Guy "NertZ" Iluz
15. Denmark Jakob "⁠jabbi⁠" Nygaard
16. Denmark Benjamin "blameF" Bremer
17. Denmark Emil "Magisk" Reif
18. Denmark Casper ⁠"cadiaN"⁠ Møller
19. Brazil Kaike "KSCERATO" Cerato
20. Canada Russel ⁠"Twistzz"⁠ Van Dulken
1. Ukraine Oleksandr "⁠s1mple⁠" Kostyliev
2. France Mathieu "⁠ZywOo⁠" Herbaut
3. Russia Dmitry "⁠sh1ro⁠" Sokolov
4. Russia Sergey "⁠Ax1Le⁠" Rykhtorov
5. Bosnia and Herzegovina Nikola "⁠NiKo⁠" Kovač
6. Latvia Helvijs "broky" Saukants
7. Russia Ilya "m0NESY" Osipov
8. Estonia Robin "ropz" Kool
9. Brazil Kaike "⁠KSCERATO⁠" Cerato
10. Denmark Martin "⁠stavn⁠" Lund
11. Canada Russel "Twistzz" Van Dulken
12. Denmark Benjamin "⁠blameF⁠" Bremer
13. Norway Håvard "⁠rain⁠" Nygaard
14. Bosnia and Herzegovina Nemanja "huNter-" Kovač
15. Latvia Mareks "⁠YEKINDAR⁠" Gaļinskis
16. Ukraine Valeriy "b1t" Vakhovskiy
17. Slovakia David "⁠frozen⁠" Čerňanský
18. Israel Lotan "Spinx" Giladi
19. Brazil Yuri "yuurih" Santos
20. Russia Dzhami "Jame" Ali
1. Ukraine Aleksandr "s1mple" Kostyliev
2. France Mathieu "⁠ZywOo⁠" Herbaut
3. Bosnia and Herzegovina Nikola "NiKo" Kovač
4. Russia Dmitry "sh1ro" Sokolov
5. Russia Sergey "⁠Ax1Le⁠" Rykhtorov
6. Kazakhstan Abay "HObbit" Khasenov
7. Russia Denis "electroNic" Sharipov
8. Latvia Mareks "YEKINDAR" Gaļinskis
9. Ukraine Valeriy "⁠b1t⁠" Vakhovskiy
10. Russia Dzhami "⁠Jame⁠" Ali
11. Denmark Nicolai "device" Reedtz
12. Bosnia and Herzegovina Nemanja "huNter-" Kovač
13. Denmark Benjamin "blameF" Bremer
14. Canada Keith "NAF" Markovic
15. Brazil Kaike "KSCERATO" Cerato
16. Denmark Martin "stavn" Lund
17. Canada Russel "Twistzz" Van Dulken
18. Estonia Robin "ropz" Kool
19. United States Jonathan "⁠EliGE⁠" Jablonowski
20. Latvia Helvijs "broky" Saukants
1. France Mathieu "⁠ZywOo⁠" Herbaut
2. Ukraine Aleksandr "⁠s1mple⁠" Kostyliev
3. Denmark Nicolai "device" Reedtz
4. Bosnia and Herzegovina Nikola "⁠NiKo⁠" Kovač
5. Russia Denis "⁠electronic⁠" Sharipov
6. Denmark Benjamin "blameF" Bremer
7. Estonia Robin "⁠ropz⁠" Kool
8. United States Jonathan "EliGE" Jablonowski
9. Denmark Peter "dupreeh" Rasmussen
10. Germany Florian "⁠syrsoN⁠" Rische
11. Denmark Emil "⁠Magisk⁠" Reif
12. Denmark Martin "⁠stavn⁠" Lund
13. Bosnia and Herzegovina Nemanja "huNter-" Kovač
14. Brazil Yuri "⁠yuurih⁠" Santos
15. Sweden Ludvig "⁠Brollan" Brolin
16. Brazil Henrique "⁠HEN1⁠" Teles
17. Sweden Freddy "KRIMZ" Johansson
18. Brazil Kaike "⁠KSCERATO⁠" Cerato
19. Australia Justin "⁠jks⁠" Savage
20. United States Vincent ⁠"Brehze⁠" Cayonte
1. France Mathieu "ZywOo" Herbaut
2. Ukraine Aleksandr "s1mple" Kostyliev
3. Denmark Nicolai "device" Reedtz
4. United States Jonathan "EliGE" Jablonowski
5. Denmark Emil "Magisk" Reif
6. Russia Denis "electronic" Sharipov
7. Canada Keith "NAF" Markovic
8. United States Vincent "Brehze" Cayonte
9. Canada Russel "Twistzz" Van Dulken
10. Estonia Robin "ropz" Kool
11. Bosnia and Herzegovina Nikola "NiKo" Kovac
12. Turkey Özgür "woxic" Eker
13. Finland Jere "sergej" Salo
14. Denmark Andreas "Xyp9x" Højsleth
15. Australia Justin "jks" Savage
16. Denmark Peter "dupreeh" Rasmussen
17. Sweden Freddy "KRIMZ" Johansson
18. Bulgaria Tsvetelin "CeRq" Dimitrov
19. Sweden Ludvig "Brollan" Brolin
20. United States Ethan "Ethan" Arnold
1. Ukraine Aleksandr "s1mple" Kostyliev
2. Denmark Nicolai "device" Reedtz
3. Bosnia and Herzegovina Nikola "NiKo" Kovac
4. Russia Denis "electronic" Sharipov
5. Denmark Peter "dupreeh" Rasmussen
6. Canada Keith "NAF" Markovic
7. Denmark Emil "Magisk" Reif
8. Denmark Lukas "gla1ve" Rossander
9. Sweden Freddy "KRIMZ" Johansson
10. Brazil Marcelo "coldzera" David
11. Slovakia Ladislav "GuardiaN" Kovács
12. Canada Russel "Twistzz" Van Dulken
13. Denmark Andreas "Xyp9x" Højsleth
14. Czech Republic Tomáš "oskar" Štastný
15. United States Jonathan "EliGE" Jablonowski
16. Finland Miikka "suNny" Kemppi
17. United States Timothy "autimatic" Ta
18. Norway Håvard "rain" Nygaard
19. Estonia Robin "ropz" Kool
20. Denmark Valdemar "valde" Bjørn Vangså
"""

# Extract names from the given segment
names = extract_and_clean_in_game_names(text_segment)

# Print extracted and cleaned names
for name in names:
    print(name)
