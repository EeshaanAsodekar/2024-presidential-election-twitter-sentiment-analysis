import pandas as pd

# Pro-Trump Tweets
pro_trump_tweets = [
    "RT @PatriotVoice: President Trump is the only one who can bring back jobs and secure our borders. #MAGA",
    "Just watched Trump's rally—feeling inspired and hopeful for America's future!",
    "Trump's policies have strengthened our economy and military. Four more years!",
    "Proud to support a leader who puts America first. Go Trump!",
    "Trump's commitment to protecting our freedoms is unmatched. Voting Trump 2024!",
    "Only Trump can fix the mess we're in. He's the strong leader we need.",
    "Supporting Trump because he puts American citizens first.",
    "Trump stands for law and order. We need him back in the White House.",
    "Economy was booming under Trump. Can't wait to have that back!",
    "Trump's vision for America aligns with mine. He's got my vote.",
    "Proud to be part of the #SilentMajority supporting Trump!",
    "Trump's foreign policy kept us safe. Time to restore that strength.",
    "No one fights for American workers like Trump does.",
    "Trump 2024—let's make America great again, again!",
    "His leadership is unparalleled. Trump has my full support.",
    "Trump's stance on border security is exactly what we need.",
    "Voting Trump to protect our constitutional rights.",
    "He brought peace deals to the Middle East. Trump knows how to lead.",
    "Under Trump, unemployment was at an all-time low. Let's get back to that.",
    "Trump's America First policies benefit all of us. He has my vote.",
    "RT @KanekoaTheGreat: NEW: Donald Trump compliments Elon Musk and SpaceX for landing their rocket this morning.",
    "They are terrified of TRUMP because he stands up for the working class.",
    "Trump will end the war and bring our troops home!",
    "The National Border Patrol Council has endorsed President Trump. That's huge!",
    "Trump is the only one who can fix the economy and secure our future.",
    "Watching Trump speak always reminds me why I support him.",
    "Trump's energy policies made us energy independent. Let's get back to that.",
    "He puts America first, always. That's why I stand with Trump.",
    "Trump's tax cuts helped my small business thrive.",
    "No one can rally a crowd like Trump. The enthusiasm is real!",
    "Trump understands the challenges facing everyday Americans.",
    "Under Trump, we had peace through strength.",
    "Trump's policies benefit all Americans, regardless of background.",
    "He respects our military and veterans. Proud to support Trump.",
    "Trump's commitment to law and order keeps our communities safe.",
    "He stands up against the establishment. Go Trump!",
    "Trump's leadership on trade deals protected American jobs.",
    "He kept his promises during his first term. Ready for his second!",
    "Trump's dedication to protecting our rights is unmatched.",
    "He has a proven track record of success. Trump 2024!",
    "Trump will make sure America remains the land of opportunity.",
    "Our nation needs strong leadership, and Trump provides that.",
    "Trump's vision for the future is exactly what we need.",
    "He never backs down from a challenge. That's a true leader.",
    "Trump's focus on American interests is refreshing.",
    "He fights for us every day. That's why I'm voting Trump.",
    "Trump's stance on education empowers parents and students.",
    "His commitment to fair trade deals helps American workers.",
    "Trump is the leader who gets things done. Let's re-elect him!",
    "He believes in the American dream and works to protect it.",
    "Trump's border policies keep our nation secure.",
    "He stands strong against adversaries. Proud to support Trump.",
    "Trump's economic policies brought prosperity. Let's continue that.",
    "He honors our history and works for a better future.",
    "Trump's dedication to freedom is why he has my vote.",
    "He puts the power back in the hands of the people.",
    "Trump's approach to foreign policy keeps us safe.",
    "He supports law enforcement and first responders.",
    "Trump's optimism about America is contagious.",
    "He defends the Constitution wholeheartedly.",
    "Trump's leadership is exactly what this country needs right now.",
    "He listens to the American people. That's why I support Trump.",
    "Trump's policies have real, positive impacts on our lives.",
    "He stands up to the mainstream media and tells it like it is.",
    "Trump's dedication to job creation is unmatched.",
    "He fights against corruption and puts America first.",
    "Trump's courage to face challenges head-on inspires me.",
    "He believes in freedom of speech and expression.",
    "Trump's support for the Second Amendment is unwavering.",
    "He has the experience to lead us forward.",
    "Trump's commitment to healthcare reform benefits us all.",
    "He champions small businesses and entrepreneurship.",
    "Trump's focus on infrastructure will rebuild our nation.",
    "He respects religious freedoms and values.",
    "Trump's actions speak louder than words.",
    "He brings a business mindset to government.",
    "Trump's strong stance on national security keeps us protected.",
    "He works tirelessly for the American people.",
    "Trump's passion for this country is evident.",
    "He isn't afraid to challenge the status quo.",
    "Trump's leadership during crises shows his capabilities.",
    "He is a voice for those who feel unheard.",
    "Trump's commitment to education reform is needed.",
    "He supports our farmers and rural communities.",
    "Trump's pro-growth policies benefit everyone.",
    "He understands the importance of energy independence.",
    "Trump's focus on fair taxation helps working families.",
    "He stands firm against illegal immigration.",
    "Trump's policies strengthen our position in the world.",
    "He is dedicated to making America prosperous again.",
    "Trump's vision aligns with traditional American values.",
    "He is the leader who will keep America great.",
]

# Pro-Harris Tweets
pro_harris_tweets = [
    "RT @KamalaHQ: Kamala Harris is the leader we need to bring unity and progress. #Harris2024",
    "Excited to vote for Harris—she stands for justice and equality!",
    "Kamala's plans for healthcare and education are exactly what this country needs.",
    "Proud to support Kamala Harris, a strong voice for change and innovation.",
    "Harris has the vision and integrity to lead us forward. #VoteHarris",
    "Kamala Harris will restore dignity and respect to the presidency.",
    "We need Harris to continue fighting for environmental justice.",
    "Voting for Harris because she cares about all Americans.",
    "Her commitment to social justice is inspiring. Harris 2024!",
    "Excited to see what Harris will accomplish in her first term!",
    "Harris's experience and compassion make her the perfect candidate.",
    "She's breaking barriers and setting examples. Go Kamala!",
    "Kamala's focus on healthcare will benefit so many families.",
    "Her plans for criminal justice reform are much needed.",
    "Harris is the leader who will unite us during these times.",
    "Proud to support a candidate who values diversity and inclusion.",
    "Kamala Harris understands the challenges we face and has solutions.",
    "She's fighting for women's rights and equality. #ImWithHer",
    "Harris's dedication to education will shape our future.",
    "Looking forward to positive change under Harris's leadership.",
    "RT @TheRickyDavila: I'll constantly say it, polls could show Kamala Harris winning by 70 points in every state and I'd still vote for her.",
    "VP Harris's plans for economic recovery are just what we need.",
    "Her leadership during crises has been commendable.",
    "Kamala's commitment to affordable education is inspiring.",
    "She stands for working families and equitable opportunities.",
    "Harris's environmental policies will protect our planet.",
    "Excited about Harris's vision for healthcare reform.",
    "She champions civil rights and fights against injustice.",
    "Kamala Harris is the future of progressive leadership.",
    "Her dedication to public service is unmatched.",
    "Harris's foreign policy experience is crucial.",
    "She brings people together across party lines.",
    "Kamala's focus on infrastructure will rebuild our communities.",
    "She supports small businesses and economic growth.",
    "Harris's stance on healthcare ensures coverage for all.",
    "Her plans for job creation will boost the economy.",
    "Proud to support a candidate who values science and facts.",
    "Kamala's approach to immigration is compassionate and fair.",
    "She advocates for voting rights and democracy.",
    "Harris's dedication to education reform is needed.",
    "Her leadership on COVID-19 response has saved lives.",
    "Excited to see Harris's initiatives on criminal justice.",
    "She fights for equal pay and workers' rights.",
    "Kamala Harris listens to the people and acts accordingly.",
    "Her policies aim to reduce income inequality.",
    "Harris's vision includes everyone, no matter their background.",
    "She stands strong on protecting Social Security and Medicare.",
    "Kamala's experience as VP prepares her for the presidency.",
    "She promotes unity and healing in our nation.",
    "Harris's focus on renewable energy is forward-thinking.",
    "Her commitment to mental health services is vital.",
    "She supports our veterans and military families.",
    "Kamala Harris is a leader we can trust.",
    "Her plans will strengthen our economy and infrastructure.",
    "Excited about Harris's proposals for gun safety.",
    "She values diplomacy and global partnerships.",
    "Harris's dedication to child care and family leave is important.",
    "She stands for LGBTQ+ rights and equality.",
    "Kamala's policies will improve healthcare access.",
    "Her vision includes criminal justice reform and fairness.",
    "Proud to vote for a candidate who champions diversity.",
    "Harris's leadership will restore America's standing in the world.",
    "She supports affordable housing initiatives.",
    "Kamala Harris understands the importance of climate action.",
    "Her focus on cybersecurity will protect our nation.",
    "She advocates for humane immigration policies.",
    "Harris's commitment to education will benefit future generations.",
    "Her plans address systemic racism and promote equity.",
    "Kamala's dedication to public health is commendable.",
    "She stands with teachers and invests in education.",
    "Harris's policies support innovation and technology.",
    "Her leadership style is inclusive and collaborative.",
    "Excited to see Harris address the challenges we face.",
    "She believes in science and will tackle climate change.",
    "Kamala Harris will bring integrity back to the presidency.",
    "Her plans for economic recovery are comprehensive.",
    "She supports criminal justice reform and ending mass incarceration.",
    "Harris's dedication to the arts and culture enriches society.",
    "She champions the rights of indigenous communities.",
    "Kamala's focus on health equity is much needed.",
    "Her leadership will guide us through challenging times.",
    "She stands for reproductive rights and women's health.",
    "Harris's experience makes her the right choice for president.",
    "Proud to support a candidate who values human rights.",
    "She will work to bridge the divides in our country.",
    "Kamala Harris is committed to fighting climate change.",
    "Her plans for infrastructure will create jobs.",
    "She advocates for fair taxation and closing loopholes.",
    "Harris's policies will strengthen our democracy.",
    "She believes in the power of education to transform lives.",
    "Excited to vote for a leader who inspires hope.",
    "Kamala's commitment to justice is unwavering.",
    "She will restore America's leadership on the global stage.",
    "Harris's dedication to public service sets her apart.",
    "Her vision for America includes everyone.",
]

# Neutral Tweets
neutral_tweets = [
    "Looking forward to seeing the debates between the candidates.",
    "It's important to research all candidates before making a decision.",
    "This election is going to be one for the history books.",
    "Both parties have a lot at stake in this upcoming election.",
    "Voter turnout this year is going to be interesting to watch.",
    "Can't wait to see the results on election night.",
    "Wondering how recent events will impact the polls.",
    "Politics aside, we need to come together as a nation.",
    "The media coverage this election cycle has been intense.",
    "Social media is playing a big role in this election.",
    "Hoping for a peaceful and fair election process.",
    "I've been getting so many political ads lately!",
    "Curious to see how third-party candidates will affect the race.",
    "The debates should provide more clarity on key issues.",
    "Voting is a civic duty—make sure you're registered!",
    "The youth vote could be a game-changer this year.",
    "Early voting starts soon—plan accordingly.",
    "Economic policies are a major focus this election.",
    "Healthcare remains a top concern for many voters.",
    "Environmental issues are finally getting more attention.",
    "This election could shape the future of the Supreme Court.",
    "Education policies are crucial for our children's future.",
    "International relations are a significant part of this election.",
    "Civil rights are at the forefront of many discussions.",
    "Campaign strategies this year are very aggressive.",
    "The role of social media in politics is fascinating.",
    "Voter education is essential for a functioning democracy.",
    "The impact of the economy on the election is undeniable.",
    "Security and defense policies are important topics.",
    "Climate change policies could sway undecided voters.",
    "The importance of mail-in voting is being highlighted.",
    "The candidates' plans for COVID-19 are under scrutiny.",
    "Tax policies are a significant concern for many.",
    "Job creation is a key issue this election cycle.",
    "The influence of endorsements is interesting to observe.",
    "Foreign policy debates are heating up.",
    "Voting rights discussions are more important than ever.",
    "Infrastructure plans are being discussed by both sides.",
    "Healthcare affordability is a major topic.",
    "Economic recovery strategies are being compared.",
    "Education reform is a hot topic this year.",
    "The impact of recent events on the election is significant.",
    "Trade policies are being debated extensively.",
    "The candidates' leadership styles are quite different.",
    "Cybersecurity is a growing concern in politics.",
    "The role of fact-checking in debates is crucial.",
    "Youth engagement in politics seems to be increasing.",
    "The importance of bipartisanship is being discussed.",
    "Voting deadlines are approaching—stay informed!",
    "The future of technology policy is at stake.",
    "Healthcare access disparities are being highlighted.",
    "The impact of endorsements from prominent figures.",
    "Economic inequality is a key issue for many voters.",
    "The significance of Supreme Court appointments.",
    "Environmental policies are influencing voter decisions.",
    "The role of the vice-presidential candidates is important.",
    "National security is a major topic this election.",
    "The influence of political action committees is notable.",
    "Voter registration numbers are climbing.",
    "The importance of local elections alongside the national one.",
    "Policy differences between the candidates are stark.",
    "Media bias is a topic of discussion among voters.",
    "The effect of polling data on campaign strategies.",
    "Debate performances can sway undecided voters.",
    "Campaign financing is under scrutiny this year.",
    "Voter suppression concerns are being raised.",
    "The impact of political advertisements on public opinion.",
    "The role of grassroots movements in the election.",
    "Election security measures are being evaluated.",
    "Public opinion polls show varying results.",
    "The significance of political endorsements.",
    "Voting methods vary by state—know your options.",
    "The candidates' stances on key issues are crucial.",
    "The impact of early voting on election outcomes.",
    "Political discussions are becoming more prevalent.",
    "The importance of staying informed during election season.",
    "Voter education campaigns are ramping up.",
    "The future direction of the country is at stake.",
    "Political rallies are drawing large crowds.",
    "The candidates' plans for education funding.",
    "Voter enthusiasm appears to be high this year.",
    "The influence of debate moderators is being discussed.",
    "Political pundits are making their predictions.",
    "Election laws are a hot topic in some states.",
    "The role of international observers in U.S. elections.",
    "Voter ID laws are being debated extensively.",
    "The impact of third-party candidates on the election.",
    "Campaign slogans are everywhere these days.",
    "Political engagement among young people is increasing.",
    "The importance of down-ballot races is being emphasized.",
    "Election night coverage plans are being announced.",
    "Polling places are preparing for Election Day.",
    "The role of political debates in shaping opinions.",
    "The candidates' views on social issues are pivotal.",
    "The potential for a close election is high.",
    "Voter turnout initiatives are in full swing.",
    "The impact of campaign speeches on voters.",
    "Election predictions vary widely this year.",
    "The significance of the electoral college is discussed.",
    "Political ads are more prevalent than ever.",
    "The role of political analysts in the media.",
    "The importance of fact-checking during campaigns.",
    "The effect of economic indicators on the election.",
    "The influence of celebrities endorsing candidates.",
    "Public policy discussions are at the forefront.",
    "The role of political debates in democracy.",
]

negative_trump_tweets = [
    "RT @IanSams: CBS: \"More than 230 doctors and health care providers call on Trump to release medical records\" \"Trump is falling concerningl…",
    "RT @RonFilipkowski: After abandoning rally goers in Coachella with one-way bus rides, Trump then puts his cultists in a steaming hot buildi…",
    "RT @JoJoFromJerz: The NYT called Trump’s batshit insane thirty nine minute long rally-stage psycho swaying to random music an “improvisatio…",
    "RT @SarahLongwell25: Watch Kamala Harris’s rally in PA and then compare it to Trump’s insane townhall-turned-boomer-musical and it’s clear…",
    "RT @Out5p0ken: Aaron is doing what MSM is not. The media needs to start showing how insane Trump is, acts, and what he says. Not doing so…",
    "RT @KamalaHQ: Trump: Whenever I go, ‘Hannibal Lecter,’ you know what I'm talking about. They say ‘Hannibal Lecter, why would he mention?’ W…",
    "RT @RonFilipkowski: The faces on half the people are just saying WTF as Trump plays music for 30 minutes. Half of the cultists are going wi…",
    "RT @kp_official_1: We are in real time feeling the ramifications of a lot of the Trump policies put in place during his term in regards of…",
    "RT @McFaul: While president, Trump actively courted dictators in Russia and North Korea. His praise of the two of them produced no concrete…",
    "RT @TVMoJoe: This is the first draft of the NYT coverage of Trump’s bizarre event. “Nothing to see here,” says the paper of record.",
    "RT @HillaryClinton: Let's be absolutely clear so that no one is confused. Trump's rhetoric has become blatantly fascist. All the warnin…",
    "RT @Scaramucci: Larry Sabato: “Trump is talking about putting General Milley in the electric chair.” How are we normalizing this? This i…",
    "RT @AdamParkhomenko: Let’s get 500 retweets as soon as possible on Kamala Harris destroying Donald Trump",
    "RT @KamalaHarris: Yesterday, I reaffirmed how we can come together and support our fellow Americans in times of crisis. Donald Trump sugge…",
    "RT @MeidasTouch: This is BIG. The Washington Post just published an article about Trump’s disastrous event tonight and discussed his declin…",
    "RT @atrupar: This entire exchange between Jake Tapper and Glenn Youngkin in which Youngkin pretends Trump didn't actually threaten to use t…",
    "RT @asweetgrace: HOLD. TRUMP. TO. THE. DEMENTIA. STANDARD. YOU. ALL. SET. FOR. BIDEN.",
    "RT @Strandjunker: Have you ever wondered why Germans didn’t do anything about Hitler? Well, the rest of the world is wondering exactly that…",
    "RT @AccountableGOP: Trump is unwell: \"If everybody gets out and votes on January 5th.\"",
    "RT @piper4missouri: You’re not crazy. You’re paying attention to the news and you are listening to the words coming out of their fascist mo…",
    "RT @EkbMary: People’s “Trump-is-the-only person-that-matters” tunnel vision has made them blind to everything else the DOJ has done.",
    "RT @eileenvan55: TO UPHOLD 'VALIDITY' OF THE RULE OF LAW & THE CONSTITUTION, Donald John Trump MUST face trial for his participation in the January 6, 2021 Insurrection",
    "RT @SarahIronside6: I have a question for MAGA. How exactly has illegal immigration ACTUALLY impacted your life? Have you ever considered t…",
    "RT @JoJoFromJerz: Now that Kamala is going to sit down with Bret Baier, surely Trump will sit down with Rachel Maddow, right?",
    "RT @IAmPoliticsGirl: Watching Trump on that stage, oblivious to what was going on around him, you know his team has been hiding his decline…",
    "RT @NickKnudsenUS: Watch Donald Trump ATTEMPT to answer this question about helping small businesses. It’s WILD.",
    "RT @davidhogg111: Our potential next president just stood for 30 minutes like an NPC, listening to 'Time To Say Goodbye' and 'Hallelujah' not s…",
    "RT @Marmel: Pretend it’s Biden, CNN, NYTimes, Washington Post and the rest. Just… cover Trump’s dementia the way you covered the other gu…",
    "RT @Bitcoin4Freedom: I can't stand Donald Trump. He is braggy, he insults people for no reason, and he is just a brutal personality. But my…",
    "RT @FrankFigliuzzi1: There, I said it: Trump's military solution for 'enemies within' is what a fascist might propose",
    "RT @OccupyDemocrats: BREAKING: Republican commentator Geraldo Rivera abandons his former friend Donald Trump and proudly endorses Kamala Ha…",
    # Additional tweets omitted for brevity
]

negative_harris_tweets = [
    "@GeraldoRivera You are a fool if you vote for Harris. Talk about a career built on lies! You are such a disappointment Geraldo. Glad Fox gave you the boot!",
    "RT @RpsAgainstTrump: Fox’s Maria Bartiromo slams Vogue magazine for the Kamala Harris Cover: “They should be ashamed of themselves that t…",
    "@KamalaHarris All you do is talk about Trump. That is proof you have nefarious motives. Your plans are too sinister to mention openly so you paint him as the menace and slide in your devilment undetected. I pray the Lord exposes the truth and casts you out.",
    "RT @TONYxTWO: “Joe Biden is a fraud! Barack Obama is a fraud! And Kamala Harris is a fraud!” “Why would we give you a promotion for doing…",
    "Imagine if Trump did something like this for WHITE MEN. Racist AF",
    "Bill Clinton accidentally admitted something by saying on the campaign trail 🤣 The open-border policies of Kamala Harris and Joe Biden are responsible for the deaths of innocent Americans like Laken Riley.",
    "RT @GuntherEagleman: 🔥BOOM: JD Vance just called out Kamala Harris for plagiarizing her book. Everyone knows Kamala isn’t smart enough to…",
    "RT @CollinRugg: JUST IN: Kamala Harris accused of plagiarizing entire sections of her 'Smart on Crime' book in an exclusive report by @realch…",
    "RT @NahBabyNah: Comments? New York Times Admits Kamala Harris Plagiarized, Claims Passages Were 'Not Serious'",
    "Kamala Harris, Mistress of Deception. Buyer beware! You can't afford this liar.",
    "RT @BreannaMorello: While campaigning for Kamala Harris, Bill Clinton admits if the Biden-Harris regime properly vetted illegal aliens, Lake…",
    "RT @AltcoinDailyio: BREAKING: BlackRock CEO Larry Fink says neither Donald Trump nor Kamala Harris is a factor for crypto price as 'we beli…",
    "RT @Travis_4_Trump: Kamala Harris descended from a Jamaican slave-owner who owned five plantations and over 200 slaves.",
    "RT @PeterMoskos: Has Harris's account been hacked? These are horribly demeaning ideas. Why would black men want to become drug dealers? Eve…",
    "RT @VDHanson: Increasingly, little if anything remains real about the Harris campaign.",
    "RT @Mompreneur_of_3: MAGA's at Lara Trump's Boat Parade yelling F*CK N*GGERS & JEWS! If you vote for the same candidate they do, you're a r…",
    "RT @Ann_Lilyflower: A must watch by EVERY DEMOCRAT voting for this bish Kamala Harris. Kamala was and will continue to be a COMMUNIST POS.",
    # Additional tweets omitted for brevity
]

# TODO: add Vance and Waltz to the labelled lists as well

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score



def create_labelled_dataset(trump_tweets:list,
                            harris_tweets:list,
                            neutral_tweets: list)->pd.DataFrame:
    '''
    Returns a labelled tweets dataset dataframe
    Columns of the dataset:
        Tweet (the tweet text)
        Label (trump or harris or neutral)
    '''
    # Create labeled DataFrames
    df_trump = pd.DataFrame(trump_tweets, columns=["Tweet"])
    df_trump["Label"] = "trump"

    df_harris = pd.DataFrame(harris_tweets, columns=["Tweet"])
    df_harris["Label"] = "harris"

    df_neutral = pd.DataFrame(neutral_tweets, columns=["Tweet"])
    df_neutral["Label"] = "neutral"

    # Concatenate the DataFrames into a single DataFrame
    df = pd.concat([df_trump, df_harris, df_neutral], ignore_index=True)

    # Shuffle the DataFrame (optional)
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def classify_tweets(tweets_labelled_df: pd.DataFrame, tweets_df: pd.DataFrame) -> pd.DataFrame:
    # Split labelled dataset into features (X) and labels (y)
    X = tweets_labelled_df['Tweet']
    y = tweets_labelled_df['Label']

    # Encode labels to numeric values for compatibility with XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_vect = vectorizer.fit_transform(X)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y_encoded, test_size=0.2, random_state=42)

    # Initialize models to test
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'XGBoost': XGBClassifier(eval_metric='mlogloss'),
        'SVM': SVC(kernel='linear', probability=True, class_weight='balanced')
    }

    # Train and evaluate each model, storing the one with the best accuracy
    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Improved output formatting
        print(f"{model_name:<20} | Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"\nBest Model: {type(best_model).__name__} with Accuracy: {best_accuracy:.2f}")

    # Use the best model to classify tweets in tweets_df
    tweets_df_vect = vectorizer.transform(tweets_df['Text'])
    predicted_labels_encoded = best_model.predict(tweets_df_vect)

    # Decode the numeric predictions back to original labels
    predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)

    # Add the predicted labels as a new column in tweets_df
    tweets_df['Label'] = predicted_labels

    return tweets_df

if __name__ == "__main__":
    # Load tweets dataset
    tweets_df = pd.read_csv("data/raw/tweets_Presidential_Election_data_Oct15_2024.csv")
    tweets_df["Text"] = tweets_df["Text"].astype(str)

    # Create the labelled tweets dataset
    trump_tweet_list = pro_trump_tweets + negative_trump_tweets
    harris_tweets_list = pro_harris_tweets + negative_harris_tweets
    neutral_tweets_list = neutral_tweets

    tweets_labelled_df = create_labelled_dataset(trump_tweet_list, harris_tweets_list, neutral_tweets_list)

    # Classify the tweets
    classified_tweets_df = classify_tweets(tweets_labelled_df, tweets_df)

    # Print results
    print(classified_tweets_df.head())
    print(tweets_labelled_df['Label'].value_counts())
    print(classified_tweets_df['Label'].value_counts())
    classified_tweets_df.to_csv('data/raw/tweets_classified.csv')