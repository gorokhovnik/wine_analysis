import pandas as pd
import numpy as np

africa = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde',
          'Central African Republic', 'Chad', 'Comoros', 'Democratic Republic of the Congo', 'Djibouti', 'Egypt',
          'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast',
          'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco',
          'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Republic of the Congo', 'Rwanda', 'Sao Tome and Principe',
          'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'Sudan', 'Swaziland', 'Tanzania',
          'The Gambia', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe']
asia = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei Darussalam', 'Cambodia',
        'China', 'Cyprus', 'East Timor', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan',
        'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal',
        'North Korea', 'Oman', 'Pakistan', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea',
        'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan',
        'Vietnam', 'Yemen']
europe = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia',
          'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland',
          'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Malta', 'Moldova', 'Monaco',
          'Montenegro', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Republic of Ireland', 'Romania', 'Russia',
          'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom',
          'Vatican City']
north_america = ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica',
                 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico',
                 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines',
                 'Trinidad and Tobago', 'United States']
oceania = ['Australia', 'Federated States of Micronesia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Nauru',
           'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu']
south_america = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru',
                 'Suriname', 'Uruguay', 'Venezuela']

red_list = ['Agiorgitiko', 'Aglianico', 'Argaman', 'Auxerrois', 'Barbera', 'Bouchet', 'Cabernet Franc',
            'Cabernet Sauvignon', 'Calabrese', 'Caladoc', 'Cannonau', 'Carignan', 'Carinena', 'Carmenere', 'Carmenère',
            'Chiavennasca', 'Corvina', 'Cot', 'Dolceto', 'Gaglioppo', 'Gamay', 'Garnacha Tinta', 'Grenache Noir',
            'Lambrusco', 'Lampia', 'Malbec', 'Marselan', 'Mataro', 'Mavrud', 'Mazuelo', 'Melnik', 'Merlot',
            'Monastrell', 'Montepulciano', 'Mourvedre', 'Nebbiolo', 'Negrette', 'Negroamaro', 'Nero dAvola',
            'Pinot Nero', 'Pinot Noir', 'Pinotage', 'Primitivo', 'Sagrantino', 'Sangiovese', 'Shiraz', 'Spana',
            'Spatburgunder', 'St George', 'Syrah', 'Tannat', 'Tempranillo', 'Tinto Fino', 'Vranac', 'Xynomavro',
            'Zinfandel']
white_list = ['Albarino', 'Aligote', 'Alvarinho', 'Chardonnay', 'Chenin Blanc', 'Cortese', 'Ermitage', 'Furmint',
              'Garganega', 'Gewürztraminer', 'Greco', 'Gros Manseng', 'Grüner Veltliner', 'Macabeo', 'Malmsey',
              'Malvasia', 'Marsanne', 'Melon de Bourgogne', 'Muscadelle', 'Muscadet', 'Muscat', 'Müller-Thurgau',
              'Petit Manseng', 'Pino Blanco', 'Pinot Blanc', 'Pinot Gridgio', 'Pinot Gris', 'Riesling',
              'Riesling-Silvaner', 'Rivaner', 'Roussanne', 'Sauvignon Blanc', 'Savagnin', 'Semillon', 'Silvaner',
              'Sylvaner', 'Tokaji', 'Tokay Pinot Gris', 'Trebbiano', 'Ugni Blanc', 'Verdejo', 'Verdelho', 'Vernaccia',
              'Viognier', 'Viura', 'Weifiburgunder']


def FE(wine):
    wine.set_index('id', inplace=True)
    wine['year'] = wine['title'].str.extract(r'[0-9]')
    wine['continent'] = np.where(wine['country'] in europe, 'europe',
                                 np.where(wine['country'] in asia, 'asia',
                                          np.where(wine['country'] in africa, 'africa',
                                                   np.where(wine['country'] in north_america, 'north america',
                                                            np.where(wine['country'] in south_america, 'south america',
                                                                     np.where(wine['country'] in oceania, 'oceania',
                                                                              None))))))
