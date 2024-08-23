import trueskill


class TrueSkill():
    def __init__(self, args, num_agents, init_draw_prob, init_mu):
        self.num_agents = num_agents
        
        
        self.env = trueskill.TrueSkill(draw_probability=init_draw_prob)
        self.ally_rating = [self.env.create_rating(mu = init_mu) for _ in range(self.num_agents)]
        self.opponent_rating = [self.env.create_rating(mu = init_mu) for _ in range(self.num_agents)]
        
    def rating(self, rank_list):
        for rank in rank_list:
            self.ally_rating, self.opponent_rating = self.env.rate(
                rating_groups = [self.ally_rating, self.opponent_rating],
                ranks = rank,
            )
            
    def reset(self):
        self.opponent_rating = self.ally_rating
        
        
        
        
        
        
        