#import ffai.ai.pathfinding as pf

from Curriculum import *


# ### CONVERTED LECTURES ### 
class GameAgainstRandom(Lecture): 
    def __init__(self): 
        super().__init__("vs. Random", 10)

    def evaluate(self, game, drive_over): 
        if drive_over:
            return LectureOutcome(self, win=(game.get_home_in_lead()==1), draw=game.get_home_in_lead()==0)
        else: 
            return None 
            
    def reset_game(self, config): 
        return get_empty_game_turn(config, turn=0)
        

# ### Lectures ### 
class Scoring(Lecture): 
    def __init__(self): 
        self.dst_mod = 9
        self.obstacle_mod = 4
        super().__init__("Score", self.dst_mod*self.obstacle_mod  -1) 
        
    def reset_game(self, config):
        game = get_empty_game_turn(config, turn=7)

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        dst_to_td = (level % self.dst_mod) +2 
        obstacle_level = (level // self.dst_mod) % self.obstacle_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup ball carrier
        p_carrier = home_players.pop() 
        move_player_within_square(game, p_carrier, dst_to_td, [1, board_y_max], give_ball=True)
        
        extra_ma = (p_carrier.position.x - 1) - p_carrier.get_ma() -1 
        p_carrier.extra_ma = max(min(extra_ma, 2), 0) 
        
        
        if obstacle_level > 0: 
            #Place it so a dodge is needed when dst_to_td is low. 
            p_obstacle = away_players.pop() 
            x =  p_carrier.position.x
            y =  p_carrier.position.y
            
            if x < 4: 
                #Force dodge or blitz 
                dx = obstacle_level - 2 
                move_player_within_square(game, p_obstacle, [x-dx, x+1], [y-1, y+1])
            else:     
                #Avoid tacklezones 
                dy = 3 - obstacle_level 
                
                move_player_within_square(game, p_obstacle, [1, x-2], [y-dy, y+dy])
                
        
        #place rest of players at random places out of the way 
        x_left   =   0
        x_right  =   p_carrier.position.x+1
        y_top    =   p_carrier.position.y-1
        y_bottom =   p_carrier.position.y+1
        
        move_players_out_of_square(game, home_players+away_players, [x_left, x_right], [y_top, y_bottom] )
        
        return game
    
    def evaluate(self, game, drive_over):
        if drive_over:
            return LectureOutcome(self, win=(game.get_home_in_lead()==1), draw=game.get_home_in_lead()==0)
        else:
            return None


class PickupAndScore(Lecture): 
    def __init__(self): 
        self.dst_mod = 7 #this number becomes steps from td 
        self.ball_mod = 4
        self.marked_ball_mod = 2
        
        super().__init__("Pickup", self.dst_mod * self.ball_mod * self.marked_ball_mod -1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        dst_to_td =     (level % self.dst_mod) +2
        ball_start =    (level // self.dst_mod) % self.ball_mod 
        marked_ball =   (level // self.dst_mod // self.ball_mod) % self.marked_ball_mod 
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        p_carrier = home_players.pop() 
        move_player_within_square(game, p_carrier, x=dst_to_td, y=[2, board_y_max -2] )
        
        scatter_ball(game, max(ball_start,1) , p_carrier.position)
        ball_pos = game.get_ball().position 
       
        
        
        #setup SURE HANDS?  
        if 0.5 < random.random():  
            p_carrier.extra_skills += [Skill.SURE_HANDS]
            
        #moves needed 
        moves_required = p_carrier.position.distance(ball_pos) + (ball_pos.x - 1 )
        extra_ma = moves_required - p_carrier.get_ma() 
        p_carrier.extra_ma = max(min(extra_ma, 2), 0) 
        
        
        #Move the ball a little bit to enable score. 
        move_ball_dx = max(moves_required - p_carrier.get_ma() -1,0) #one GFI ok  
        game.get_ball().position.x -= move_ball_dx
        if game.get_ball().position == p_carrier.position: 
            game.get_ball().position.x -= 1 #Make sure it's not moved unto the carrier 
        
        
        ball_pos = game.get_ball().position 
        p_pos = p_carrier.position 
        #Mark the ball
        if marked_ball>0: 
            marker = away_players.pop()
            game.move(marker, get_boundary_square(game, 1, ball_pos ) )
            marker.state.up = random.random() < 0.7
            
            if 0.5 < random.random():  
                p_carrier.extra_skills += [Skill.DODGE]
        
        
        #place rest of players at random places out of the way 
        x_left   =   0
        x_right  =   max( p_pos.x, ball_pos.x)+1
        y_top    =   min( p_pos.y, ball_pos.y)-1
        y_bottom =   max( p_pos.y, ball_pos.y)+1
        
        p_used = 1 - level/(self.max_level)
        
        move_players_out_of_square(game, home_players, [x_left, x_right], [y_top, y_bottom],p_used=p_used)
        move_players_out_of_square(game, away_players, [x_left, x_right], [y_top, y_bottom] )
        
        if ball_start==0: 
            game.set_available_actions() 
            a = Action(action_type=ActionType.START_MOVE, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
    
    def training_done(self, game): 
        training_complete = self.turn  !=  game.state.home_team.state.turn
        training_outcome = game.state.home_team.state.score > 0 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 2/6
        
class PassAndScore(Lecture): 
    def __init__(self, handoff = True): 
        self.pass_dist_mod = 6
        self.score_dist_mod = 7
        self.noise_mod = 3
        
        self.handoff = handoff
        
        max_level = self.pass_dist_mod * self.score_dist_mod * self.noise_mod -1
        
        if handoff: 
            super().__init__("Handoff to Score", max_level) 
        else:
            super().__init__("Pass to Score", max_level)    
            
    def _reset_lecture(self, game): 
        
        assert game.is_pass_available()
        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        extra_pass_dist = 1 if self.handoff else 4
        
        level = self.get_level()        
        noise      = (level %  self.noise_mod)
        dist_pass  = (level // self.noise_mod) % self.pass_dist_mod + extra_pass_dist 
        dist_to_td = (level // self.noise_mod // self.pass_dist_mod) % self.score_dist_mod +1 #1 = start in td zone
        
        
        #get players 
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup scorer 
        p_score = home_players.pop() 
        p_score_x = dist_to_td
        p_score_y = randint(2, board_y_max -1 )
        p_score.extra_skills = [] if random.random() < 0.5 else [Skill.CATCH]
        game.move(p_score, Square( p_score_x, p_score_y) )
        
        #setup passer
        p_pass  = home_players.pop() 
        #p_pass.extra_skills = [] if random.random() < 0.5 else [Skill.PASS] #Pass skill not implemeted
        
        p_pass_x = p_score_x + dist_pass 
        dx = abs(p_pass_x - p_score_x) 
        move_player_within_square(game, p_pass, x=p_pass_x, y=[p_score_y-dx, p_score_y+dx], give_ball=True )
        
        #setup passer movement left 
        max_movement = p_pass.get_ma() + 2 #add two to remove GFIs
        p_pass.state.moves  = max_movement
        
        
        if self.handoff: 
            if dx > 1: 
                #make sure passer can't score but can reach scorer 
                to_not_score= p_pass_x -1 #double gfi to score is ok. 
                to_handoff = dx #double gfi to score is ok.
                assert to_handoff <= to_not_score 
                
                p_pass.state.moves -= randint(to_handoff, min(to_not_score, max_movement) ) 
            
        else: #PassAction  
            if noise > 0: 
                #make sure passer can't reach scorer: 
                to_not_handoff = dx-2 
                p_pass.state.moves  = p_pass.get_ma() + 2 #add two to remove GFIs 
                p_pass.state.moves -= randint(0, min(to_not_handoff, max_movement) ) 
                
        assert 0 <= p_pass.state.moves 
        
        
        if level == 0 or (noise == 0 and random.random() < 0.2) : 
            # Start the pass/handoff action 
            
            action_type = ActionType.START_HANDOFF if self.handoff else ActionType.START_PASS
            a = Action(action_type=action_type, position = p_pass.position, player = p_pass )
            game.step(a)
        
        if True: 
            x_min = 0
            x_max = max(p_score.position.x, p_pass.position.x)+1
            
            y_min = min(p_score.position.y, p_pass.position.y)-1
            y_max = max(p_score.position.y, p_pass.position.y)+1
            
            if noise <= 2: 
                p_used = 1-noise/2
                p_down = 1-noise/2
            else:
                p_used = 0
                p_down = 0
            
            move_players_out_of_square(game, away_players, [x_min, x_max], [y_min , y_max], p_down=p_down )
            move_players_out_of_square(game, home_players, [x_min, x_max], [y_min , y_max], p_used=p_used, p_down=p_down )
            
            self.turn = deepcopy(game.state.home_team.state.turn)  
            
        if False: 
            print("pass moves: {}/{}".format(p_pass.state.moves, p_pass.get_ma() ))
        
    def training_done(self, game): 
        training_complete = self.turn  !=  game.state.home_team.state.turn
        training_outcome = game.state.home_team.state.score > 0 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 2/6 


#TODO: When blitz is needed. Setup used assists and start the blitz action.         
class BlockBallCarrier(Lecture): 
    
    
    def __init__(self): 
        self.challenge_level = 5
        #self.obstacle_mod = 4
        super().__init__("BlockBall", self.challenge_level  -1, delta_level=0.05) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        challenge = (level % self.challenge_level) 
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup ball carrier
        p_carrier = away_players.pop() 
        move_player_within_square(game, p_carrier, [2, board_x_max-1], [2, board_y_max-1], give_ball=True)
        
        x = p_carrier.position.x
        y = p_carrier.position.y 
        
        if challenge < 5:
            if   challenge == 0: p_on_ball = 5
            elif challenge == 1: p_on_ball = 2
            elif challenge == 2: p_on_ball = 2
            elif challenge == 3: p_on_ball = 3
            elif challenge == 4: p_on_ball = 2
            
            for _ in range(p_on_ball):
                if len(home_players) == 0: break 
                
                p = home_players.pop()
                if challenge <  3: move_player_within_square(game, p, [x-1,x+1], [y-1, y+1])
                if challenge >= 3: game.move(p, get_boundary_square(game, challenge-1, p_carrier.position))  
                
                if random.random() < 0.4: p.state.up = False 
 
            
        #place rest of players out of the way 
        
        move_players_out_of_square(game, home_players+away_players, [x-3, x+3], [y-3, y+3] )
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
    
    def training_done(self, game): 
        
        carrier = game.get_ball_carrier()
        
        
        training_outcome = game.state.home_team.state.score > 0 or carrier not in get_away_players(game)
        
        training_complete = self.turn  !=  game.state.home_team.state.turn or training_outcome
        
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0 

class CrowdSurf(Lecture): 
    def __init__(self): 
        #noise = 0 blitz/block started  
        #victim with ball 
        #blitz steps needed 
        
        self.noise_mod = 2
        self.with_ball_mod = 2
        self.steps_mod = 6
        
        super().__init__("Crowd Surf",self.noise_mod * self.with_ball_mod * self.steps_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise      = (level %  self.noise_mod)
        with_ball  = (level // self.noise_mod) % self.with_ball_mod 
        steps      = (level // self.noise_mod // self.with_ball_mod) % self.steps_mod
        
        blitz_steps = max(steps-1, 0)
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup victim
        p_victim = away_players.pop() 
        y = random.choice( [1, board_y_max] ) 
        move_player_within_square(game, p_victim, [2, board_x_max-1], y, give_ball= with_ball>0 )
        x = p_victim.position.x
        y = p_victim.position.y 
        
        dy = 1 if y == 1 else -1 
        
        #setup blocker 
        p_blocker = home_players.pop() 
        
        if blitz_steps == 0: 
            move_player_within_square(game, p_blocker, x=x, y=y+dy)
            assert p_blocker.position.x == p_victim.position.x 
        else:
            move_player_within_square(game, p_blocker, x=[x-blitz_steps,x+blitz_steps], y=y+ dy +dy*blitz_steps )
        
        
        #setup assists 
        assists = randint(0,1)
        for _ in range(assists): 
            move_player_within_square(game, home_players.pop(), x=[x-1, x+1], y=[y,y+dy], p_down= (noise>0)*level/self.max_level)
            
        
        #add strenth? 
        if game.num_block_dice(p_blocker, p_victim) < 0: 
            p_blocker.extra_st = 1
            
        
        if noise == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_BLOCK if steps==0 else ActionType.START_BLITZ
            a = Action(action_type=action_type, position = p_blocker.position, player = p_blocker )
            game.step(a)
            
       
        p_used = 1-level/self.max_level
        
        dr = blitz_steps +3 
        move_players_out_of_square(game, home_players, [x-dr, x+dr], [y-dr, y+dr], p_used=p_used )
        move_players_out_of_square(game, away_players, [x-dr, x+dr], [y-dr, y+dr])
        
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.len_away_team = len( get_away_players(game) ) 
    
        self.blocker = p_blocker
        self.victim = p_victim 
        
        
        
    def training_done(self, game): 
        
        training_outcome = self.len_away_team > len( get_away_players(game) ) 
        
        training_complete = self.turn  !=  game.state.home_team.state.turn or training_outcome
        
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 1/6

class PreventScore(Lecture): 
    def __init__(self, home_defence, reverse_agent_play_first = False , debug = False): 
        #Reverse = True - used to evaluate that the agent can score if nothing is done from the setup.
        #reverse_play_first = True - Let away agent do a turn first then let home try to score. TODO! 
        assert not reverse_agent_play_first #not implemented 
        
        self.home_defence = home_defence
        self.debug = debug 
        
        self.noise_mod = 2 
        self.scatter1_mod = 5 
        self.scatter2_mod = 5 
        self.offence_support_mod = 4
        
        max_level = self.noise_mod * self.scatter1_mod * self.scatter2_mod * self.offence_support_mod -1
        
        name = f"Prevent {'(reverse)' if not home_defence else ''}" 
        
        delta_level = 0.1 if home_defence else 0.4 
        
        super().__init__(name, max_level, delta_level= delta_level)
        
    def _reset_lecture(self, game): 
        
        if self.home_defence and not self.debug: 
            if game.away_agent.name.find("selfplay") < 0: 
                print(f"expected away agent name 'selfplay*', got '{game.away_agent.name}'")
            
                assert False 
        
        
        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        home_on_defence = self.home_defence # have to pick from argument to init() 
        noise           = level % self.noise_mod 
        scatter_1       = (level // self.noise_mod) % self.scatter1_mod
        scatter_2       = (level // self.noise_mod // self.scatter1_mod) % self.scatter2_mod
        offence_support = (level // self.noise_mod // self.scatter1_mod // self.scatter2_mod) % self.offence_support_mod
        
        if level%2==0:
            temp = scatter_1 
            scatter_1 = scatter_2 
            scatter_2 = temp 
        
        #get players 
        if not home_on_defence: 
            #0 - home team threatens score, 
            offence = get_home_players(game)
            defence = get_away_players(game)
             
        else: 
            #1 - away team threatens score 
            defence = get_home_players(game)
            offence = get_away_players(game)
            #swap needed 
            
        
        #setup ball carrier  
        p_carrier = offence.pop() 
        move_player_within_square(game, p_carrier, x=[3, 9], y=[2, board_y_max-1], give_ball=False)
        extra_ma = (p_carrier.position.x - 1) - p_carrier.get_ma() 
        p_carrier.extra_ma = max(min(extra_ma, 2), 0)
        
        ball_x = p_carrier.position.x
        ball_y = p_carrier.position.y 
        
        #setup players intended for action 
        
        marker_up_pos   = Square(ball_x-1, max(ball_y-1, 1) )        
        marker_down_pos = Square(ball_x-1, min(ball_y+1, board_y_max) )        
        
        guy1 = defence.pop()
        guy2 = defence.pop()
        
        game.move(guy1, get_boundary_square(game, scatter_1, marker_up_pos))
        game.move(guy2, get_boundary_square(game, scatter_2, marker_down_pos) )
        
        guy1.state.up = guy1.position.distance(p_carrier.position) > 2
        guy2.state.up = guy2.position.distance(p_carrier.position) > 2
        
        #setup rest of screen (in state used 
        p_used_defence = (1-level/ self.max_level) * home_on_defence
        p_used_offence = 0 

        upwards_y = ball_y - 4
        downwards_y = ball_y + 4
        
        while upwards_y > 0: #Potential special case at equal 0. 
            move_player_within_square(game, defence.pop(), x=[ball_x-2, ball_x+2], y = upwards_y, p_used=p_used_defence)
            upwards_y -= 4 
        
        while downwards_y < board_y_max+1: #Potential special case at equal to board_y_max +1 
            move_player_within_square(game, defence.pop(), x=[ball_x-2, ball_x+2], y = downwards_y, p_used=p_used_defence) 
            downwards_y += 4 
        
        #setup offensive support
        for _ in range(offence_support): 
            move_player_within_square(game, offence.pop(), x=[ball_x-3,ball_x+3], y=[ball_y-3,ball_y+3] )

        
        #setup other players randomly 
        move_players_out_of_square(game, defence, x=[0, ball_x + 5], y=[0,20], p_used = p_used_defence) #home 
        move_players_out_of_square(game, offence, x=[0, ball_x + 5], y=[0,20], p_used = p_used_offence) 
            
        
        if home_on_defence: 
            swap_game(game) #flips the board
            if level == 0 or (noise == 0 and random.random() < 0.5): 
                #set_trace() 
                game.set_available_actions() 
                action_type = ActionType.START_MOVE
                a = Action(action_type=action_type, position = guy1.position, player = guy1)
                game.step(a)
        else: 
            pass 
            
        game.get_ball().move_to(p_carrier.position)
        game.get_ball().is_carried = True 
        #Log turn 
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.opp_turn = deepcopy(game.state.away_team.state.turn)  
        
        
        
    def training_done(self, game): 
        #Play until end of drive. 
        
        if self.home_defence: 
            training_complete = self.turn  !=  game.state.home_team.state.turn
            training_outcome = game.state.away_team.state.score == 0 
            return training_complete, training_outcome
        else: 
            training_complete = self.turn  !=  game.state.home_team.state.turn
            training_outcome = game.state.home_team.state.score > 0 
            return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0
        
class ChooseBlockDie(Lecture): 
    action_types = [    ActionType.SELECT_ATTACKER_DOWN,
                        ActionType.SELECT_BOTH_DOWN,
                        ActionType.SELECT_PUSH,
                        ActionType.SELECT_DEFENDER_STUMBLES,
                        ActionType.SELECT_DEFENDER_DOWN]
    
    def __init__(self): 
        self.opp_skills = [ [], [Skill.BLOCK], [Skill.DODGE]]  
        self.own_skills = [ [Skill.BLOCK], [] ]  
        
        self.dice_mod = 3
        self.opp_skills_mod = len(self.opp_skills)
        self.own_skills_mod = len(self.own_skills)
                              
        #self.obstacle_mod = 4
        super().__init__("Choose Die", self.dice_mod * self.opp_skills_mod * self.own_skills_mod  -1, delta_level=0.05) 
    
    def _reset_lecture(self, game): 
        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        self.dices    =  3-(level % self.dice_mod) 
        blocker_skill =    (level // self.dice_mod) % self.own_skills_mod 
        victim_skill  =    (level // self.dice_mod // self.own_skills_mod)  % self.opp_skills_mod 
        
        blocker_team    = get_home_players(game)
        victim_team     = get_away_players(game) 
        
        victim = victim_team.pop() 
        move_player_within_square(game, victim, x = [2,board_x_max-1], y = [2, board_y_max-1], give_ball=random.random() < 0.5)
        x = victim.position.x
        y = victim.position.y
        
        blocker = blocker_team.pop() 
        move_player_within_square(game, blocker, x = [x-1,x+1], y = [y-1, y+1])
        
        
        #Setup skills
        if random.random() < 0.8: 
            blocker.extra_skills = self.own_skills[blocker_skill]
        if random.random() < 0.8: 
            victim.extra_skills = self.opp_skills[victim_skill]
        
        #setup assists if needed for two die 
        target_str = victim.get_st()  + 1 + victim.get_st()*(self.dices==3)
        blocker_assists  = target_str - blocker.get_st() 
        for _ in range(blocker_assists): 
            move_player_within_square(game, blocker_team.pop(), x = [x-1,x+1], y = [y-1, y+1], p_used=1)
        
        #Setup rest of players: 
        move_players_out_of_square(game, blocker_team, [x-4, x+4], [y-4, y+4], p_used=1)
        move_players_out_of_square(game, victim_team, [x-4, x+4], [y-4, y+4])
        
        #Randomly place ball 
        ball_pos = Square( randint(1,board_x_max), randint(1,board_y_max)) 
        game.get_ball().move_to( ball_pos ) 
        game.get_ball().is_carried = game.get_player_at(ball_pos) is not None  
                     
        game.set_available_actions()
        a = Action(action_type=ActionType.START_BLOCK, position = blocker.position, player = blocker )
        game.step(a)
        a = Action(action_type=ActionType.BLOCK, position = victim.position, player = victim)
        game.step(a)
            
        self.actions = [a.action_type for a in game.get_available_actions() ] 
        
        assert True in [a in self.actions for a in ChooseBlockDie.action_types]
           
        self.victim = victim
        self.blocker = blocker 
        
        assert game.state.active_player ==  self.blocker
        
#    def get_opp_actor(self): 
#        return ChooseBlockDie.BlockBot()
    
    def training_done(self, game): 
        
        training_complete = game.state.active_player != self.blocker
        
        training_outcome = (not self.victim.state.up) and self.blocker.state.up 
        
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        
        # if ActionType.SELECT_DEFENDER_DOWN in self.actions: 
            # return 0 
        
        # elif ActionType.SELECT_DEFENDER_STUMBLES in self.actions: 
            # return 0
            
        # elif ActionType.SELECT_BOTH_DOWN and (not self.victim.has_skill(Skill.BLOCK) ) and self.blocker.has_skill(Skill.BLOCK): 
            # return 0 
            
        # else: 
            # return 1 
        level = self.get_level()        
        self.dices    =  3-(level % self.dice_mod) 
        
        if self.dices == 1: 
            return 0.5
        elif self.dices == 2: 
            return 0.25
        else: #self.dices == 3: 
            return 0
        
class PlayAgentLecture(Lecture): 
    def __init__(self,name, agent): 
        super().__init__(name, 10)
        self.agent = agent 
    def get_opp_actor(self): 
        return self.agent 
    def lec_reset_required(self): 
        return False 
    def training_done(self, game): 
        return  game.state.game_over, (game.state.away_team.state.score <= game.state.home_team.state.score) 
    def is_full_game_lect(self): return True 
    
class PickupKickoffBall(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Kickoff pickup", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup LoS oppenent
        for _ in range( randint(3,6)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=13, y=[5,11])
        
        #setup rest of opponents 
        move_players_out_of_square(game, away_players, x=[12, 50], y=[0, 20])
        
        
        #get ball carrier
        p_carrier = home_players.pop() 
        if level / self.max_level < random.random():  
            p_carrier.extra_skills = [Skill.SURE_HANDS]
        
        #setup LoS own team
        p_used = 1 - level / (self.max_level)
        p_used = max(1, p_used)
        
        for _ in range( randint(3,6)): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=14, y=[5,11], p_used=p_used)
        
        #setup rest of team 
        for _ in range(len(home_players)): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=[15,19], y=[1,board_y_max], p_used=p_used)
        
        #setup ball 
        center_square = Square(20,8)
        scatter_ball(game, max(1,noise), center_square)
        
        #setup ball carrier
        
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game): 
        training_outcome  = game.get_ball().is_carried #game.get_ball_carrier() in  get_home_players(game) 
        training_complete = training_outcome or self.turn != game.state.home_team.state.turn or self.carrier.state.used 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
    
class PlayScriptedBot(PlayAgentLecture): 
    def __init__(self): 
        super().__init__("Scripted bot", ffai.make_bot('scripted'))

class PlayRandomBot(PlayAgentLecture): 
    def __init__(self): 
        super().__init__("Random bot", RandomBot("Random"))

        
class Lecture1(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 1", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [[16,10],[11,3],[12,5],[14,7],[12,9],[9,9],[8,12],[13,13],[14,13],[14,14]] #1st one is the ballp_carrier
        away_coords = [[10,3],[11,5],[13,6],[10,8],[12,8],[13,8],[8,9],[9,12],[7,12],[7,14]]

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:

        #setup ballp_carrier, change home to away if it's red team
        p_carrier = home_players.pop()
        move_player_within_square(game, p_carrier, x=home_coords[0,0], y=home_coords[0,1], give_ball=True)

        #setup opponents
        for a in range(len(away_players)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1])
        
        #setup own team
        for _ in range(len(home_players)-1): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a+1,0], y=home_coords[a+1,1]) 
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
        

class Lecture3(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 3", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [[11,14],[13,11],[14,6],[14,12],[14,13],[15,8],[15,9],[15,13]] #1st one is the ball p_carrier
        away_coords = [[10,8],[10,13],[12,12],[13,6],[14,7],[14,10],[14,11],[15,3],[15,4]]

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:

        #setup ball p_carrier, change home to away if it's red team
        p_carrier = home_players.pop()
        move_player_within_square(game, p_carrier, x=home_coords[0,0], y=home_coords[0,1], give_ball=True)

        #setup opponents
        for a in range(len(away_players)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1])
        
        #setup own team
        for a in range(len(home_players)-1): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a+1,0], y=home_coords[a+1,1]) 
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
    

class Lecture4(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 4", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game) 
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [ [10,3],[11,2],[11,5],[13,4],[13,6],[16,2],[17,3],[17,4] ] #1st one is the ball p_carrier #TODO czerwony
        away_coords = [ [9,2],[9,3],[11,3],[13,2], [15,3],[16,5] ]                                    #TODO niebieski

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:

        #setup ball p_carrier, change home to away if it's red team
        p_carrier = home_players.pop()    #TODO
        move_player_within_square(game, p_carrier, x=home_coords[0,0], y=home_coords[0,1], give_ball=True)

        #setup opponents
        for a in range(len(away_players)-1): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1]) #TODO a+1 -> ten kto ma piłke
        
        #setup own team
        for a in range(len(home_players)): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a+1,0], y=home_coords[a+1,1])  #TODO
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
    

class Lecture5(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 5", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [[10,13],[10,10],[14,12],[16,9],[17,9]] #1st one is the ball p_carrier
        away_coords = [[9,9],[11,8],[11,12],[15,12],[15,13],[16,12],[17,8],[18,13]]

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:

        #setup ball p_carrier, change home to away if it's red team
        p_carrier = home_players.pop()
        move_player_within_square(game, p_carrier, x=home_coords[0,0], y=home_coords[0,1], give_ball=True)

        #setup opponents
        for a in range(len(away_players)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1])
        
        #setup own team
        for a in range(len(home_players)-1): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a+1,0], y=home_coords[a+1,1]) 
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
    

class Lecture6(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7

        self.enemies_on_board = 0

        super().__init__("Lecture 6", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game) 
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [ [14,6],[10,4],[11,2],[11,3],[12,4],[12,6],[13,5],[13,7],[13,10],[13,13],[15,7] ] #1st one is the ball p_carrier #TODO czerwony
        away_coords = [ [9,2],[10,2],[10,5],[10,6],[10,7],[11,5],[11,6],[11,13][12,5],[12,9],[12,11] ]                                    #TODO niebieski

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:

        #setup ball p_carrier, change home to away if it's red team
        p_carrier = home_players.pop()    #TODO
        move_player_within_square(game, p_carrier, x=home_coords[0,0], y=home_coords[0,1], give_ball=True)

        #setup opponents
        for a in range(len(away_players)-1): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1]) #TODO a+1 -> ten kto ma piłke
        
        #setup own team
        for a in range(len(home_players)): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a+1,0], y=home_coords[a+1,1])  #TODO
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )

        self.enemies_on_board = 0
        apl = get_away_players(game)
        for p in apl:
            if p.position == None:
                self.enemies_on_board += 1
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        enemies = 0
        apl = get_away_players(game)
        for p in apl:
            if p.position == None:
                enemies += 1

        training_outcome  = enemies <= (self.enemies_on_board - 2) 
        training_complete = training_outcome or self.turn != game.state.home_team.state.turn or self.carrier.state.used 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
    

class Lecture7(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 7", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [[14,10],[14,3],[14,8],[14,9],[14,11],[14,12],[14,14],[15,9],[20,8]] #1st one is the ball p_carrier
        away_coords = [[13,5],[13,8],[13,11]]

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:

        #setup ball p_carrier, change home to away if it's red team
        p_carrier = home_players.pop()
        move_player_within_square(game, p_carrier, x=home_coords[0,0], y=home_coords[0,1], give_ball=True)

        #setup opponents
        for a in range(len(away_players)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1])
        
        #setup own team
        for a in range(len(home_players)-1): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a+1,0], y=home_coords[a+1,1]) 
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
    

class Lecture8(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 8", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game) 
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [ [13,6],[13,10],[14,6],[14,7],[14,9],[15,6],[15,8],[22,9] ] #1st one is the ball p_carrier #TODO czerwony
        away_coords = [ [13,7],[13,9],[13,8],[3,2],[3,4],[3,6],[3,8],[3,10],[3,12],[3,14],[4,8] ]                                    #TODO niebieski

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:

        #setup ball p_carrier, change home to away if it's red team


        #setup opponents
        for a in range(len(away_players)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1]) #TODO a+1 -> ten kto ma piłke
        
        #setup own team
        for a in range(len(home_players)): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a,0], y=home_coords[a,1])  #TODO
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        put(game,ball,[23,10])

        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
    

class Lecture10(Lecture):
    def __init__(self):

        self.action_started_mod = 2
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 10", self.dst_mod * self.noise_mod - 1, delta_level=0.1)

    def _reset_lecture(self, game):

        # ### CONFIG ### #
        board_x_max = len(game.state.pitch.board[0]) - 2
        board_y_max = len(game.state.pitch.board) - 2

        # Level configuration
        level = self.get_level()
        noise = (level % self.noise_mod)
        action_started = (level // self.noise_mod) % self.action_started_mod
        distance = (level // self.noise_mod //
                    self.action_started_mod) % self.dst_mod

        home_players = get_home_players(game)
        away_players = get_away_players(game)

        # inserting coords for players according to lectures
        home_coords = [[15, 12], [13, 7], [13, 9], [13, 12], [13, 13], [14, 7], [14, 8], [
            14, 9], [15, 13], [16, 11]]  # 1st one is the ball p_carrier #TODO czerwony
        away_coords = [[11, 6], [12, 6], [12, 7], [12, 9], [11, 12], [
            12, 13], [14, 11], [14, 12], [15, 14]]  # TODO niebieski

        # give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:
        p_carrier = home_players.pop()  # TODO
        move_player_within_square(
            game, p_carrier, x=home_coords[0, 0], y=home_coords[0, 1], give_ball=True)

        # setup ball p_carrier, change home to away if it's red team
        p_carrier = home_players.pop()  # TODO
        move_player_within_square(
            game, p_carrier, x=home_coords[0, 0], y=home_coords[0, 1], give_ball=True)

        # setup opponents
        for a in range(len(away_players)):
            p = away_players.pop()
            # TODO a+1 -> ten kto ma piłke
            move_player_within_square(
                game, p, x=away_coords[a, 0], y=away_coords[a, 1])

        # setup own team
        for a in range(len(home_players)-1):
            p = home_players.pop()
            move_player_within_square(
                game, p, x=home_coords[a+1, 0], y=home_coords[a+1, 1])  # TODO
        # REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL

        if noise == 0:
            move_player_within_square(
                game, p_carrier, center_square.x, center_square.y)
        else:
            game.move(p_carrier, get_boundary_square(
                game, 1+distance, game.get_ball().position))

        if p_carrier.position.x < 15:
            move_player_within_square(game, p_carrier, x=[15, 18], y=[
                                      p_carrier.position.y-1, p_carrier.position.y+1])

        if action_started == 0:
            game.set_available_actions()
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type,
                       position=p_carrier.position, player=p_carrier)
            game.step(a)

        self.turn = deepcopy(game.state.home_team.state.turn)
        self.carrier = p_carrier

    def training_done(self, game):
        pos_x = self.carrier.x
        pos_y = self.carrier.y

        top_left = False
        top_right = False
        bottom_left = False
        bottom_right = False

        home_players = get_home_players(game)

        for player in home_players:
            pp = player.position

            if (pp.x == pos_x - 1) and (pp.y == pos_y - 1):
                top_left = True
            if (pp.x == pos_x + 1) and (pp.y == pos_y - 1):
                top_right = True
            if (pp.x == pos_x - 1) and (pp.y == pos_y + 1):
                bottom_left = True
            if (pp.x == pos_x + 1) and (pp.y == pos_y + 1):
                bottom_right = True

        training_outcome = top_left and top_right and bottom_left and bottom_right
        training_complete = training_outcome or self.turn != game.state.home_team.state.turn or self.carrier.state.used
        return training_complete, training_outcome

    def allowed_fail_rate(self):
        return 0


class Lecture11(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 11", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [[13,6],[13,9],[14,6],[14,7],[14,9],[15,6],[15,7],[15,8]]
        away_coords = [[10,2],[11,3],[10,5],[11,6],[11,10],[10,11],[11,13],[10,14],[13,7],[13,8],[14,8]]

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:
        
        #setup opponents
        for a in range(len(away_players)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1])
        
        #setup own team
        for a in range(len(home_players)): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a,0], y=home_coords[a,1]) 
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        put(game,ball,[7,8])
               
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      



class Lecture13(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 13", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [[6,12],[7,11],[8,13],[9,13],[12,10],[12,11]]
        away_coords = [[8,9],[2,10],[3,10]]

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:
        
        #setup opponents
        for a in range(len(away_players)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1])
        
        #setup own team
        for a in range(len(home_players)): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a,0], y=home_coords[a,1]) 
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        put(game,ball,[8,5])
               
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      


class Lecture14(Lecture): 
    def __init__(self): 
        
        self.action_started_mod = 2 
        self.dst_mod = 6
        self.noise_mod = 7
        super().__init__("Lecture 14", self.dst_mod * self.noise_mod  -1, delta_level=0.1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise           = (level % self.noise_mod) 
        action_started  = (level // self.noise_mod) % self.action_started_mod
        distance        = (level // self.noise_mod // self.action_started_mod) % self.dst_mod
        
        home_players = get_home_players(game) 
        away_players = get_away_players(game)

        #inserting coords for players according to lectures 
        home_coords = [ [14,5],[14,6],[14,7],[14,8],[14,9],[14,10],[14,12],[13,6],[13,7],[13,8],[22,8] ] #1st one is the ball p_carrier #TODO czerwony
        away_coords = [ [12,4],[12,5],[12,6],[12,7],[12,9],[12,10],[12,11],[12,12],[13,7],[13,8],[13,9] ]                                    #TODO niebieski

        #give ball to certain player using:  give_ball=True przy move_player_within_square, best to do is to set him as 1st one and use this:

        #setup ball p_carrier, change home to away if it's red team
        p_carrier = home_players.pop()    #TODO
        move_player_within_square(game, p_carrier, x=home_coords[0,0], y=home_coords[0,1], give_ball=True)

        #setup opponents
        for a in range(len(away_players)): 
            p = away_players.pop() 
            move_player_within_square(game, p, x=away_coords[a,0], y=away_coords[a,1]) #TODO a+1 -> ten kto ma piłke
        
        #setup own team
        for a in range(len(home_players)): 
            p = home_players.pop() 
            move_player_within_square(game, p, x=home_coords[a,0], y=home_coords[a,1])  #TODO
        #REMEMBER TO CHANGE IF IT'S NOT OUR PLAYER HOLDING THE BALL
        
        put(game,ball,[20,10])

        
        if noise == 0: 
            move_player_within_square(game, p_carrier, center_square.x, center_square.y) 
        else:    
            game.move(p_carrier, get_boundary_square(game, 1+distance, game.get_ball().position)) 
        
        if p_carrier.position.x < 15: 
            move_player_within_square(game, p_carrier, x=[15,18], y=[p_carrier.position.y-1, p_carrier.position.y+1] )
        
        if action_started == 0: 
            game.set_available_actions() 
            action_type = ActionType.START_MOVE
            a = Action(action_type=action_type, position = p_carrier.position, player = p_carrier )
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.carrier = p_carrier
    
    def training_done(self, game):
        training_outcome = False
        home_players = get_home_players(game)
        
        for player in home_players:
            if game.is_touchdown(player):
                training_outcome = True

        training_complete = training_outcome or self.turn != game.state.home_team.state.turn
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0      
            
    
        
    
    
if False:        
    # class Caging(Lecture): 
        # def __init__(self): 
            # self.distance_to_cage = 6
            # self.cage_setup_level = 6
            # super().__init__("Caging", self.distance_to_cage * self.cage_setup_level -1) 
        
        # def _reset_lecture(self, game): 
            
     #       ### CONFIG ### # 
            # board_x_max = len(game.state.pitch.board[0]) -2  
            # board_y_max = len(game.state.pitch.board) -2
        
    #        Level configuration 
            # level = self.get_level()        
            # distance_to_cage = (level % self.distance_to_cage) +1 
            # cage_setup_level = (level // self.distance_to_cage) % self.cage_setup_level 
            
    #        get players 
            # home_players = get_home_players(game)
            
    #        setup cage 
            # cage_x = randint(4, board_x_max -1 - distance_to_cage) 
            # cage_y = randint(2, board_y_max -1) 
            
            # ps = [home_players.pop() for _ in range(4) ]
            # game.move(ps[0], Square( cage_x+1, cage_y+1) )
            # game.move(ps[1], Square( cage_x+1, cage_y-1) )
            # game.move(ps[2], Square( cage_x-1, cage_y+1) )
            # game.move(ps[3], Square( cage_x-1, cage_y-1) )
            
            # for i in range(4):
                # if i >= cage_setup_level: 
                    # ps[i].state.used = True 
            
            # if cage_setup_level >= 5:
                # move_player_within_square(game, ps[2], [cage_x-4, cage_y+1], [cage_x-1, cage_y+4])
                # ps[2].state.up = False
                
            # if cage_setup_level >= 6:
                # move_player_within_square(game, ps[3], [cage_x-4, cage_y-4], [cage_x-1, cage_y-1])
                # ps[3].state.up = False
                
            # p_carrier = home_players.pop() 
            # move_player_within_square(game, p_carrier, [cage_x, cage_y-distance_to_cage], [cage_x+distance_to_cage, cage_y+distance_to_cage])
            # game.get_ball().move_to( p_carrier.position ) 
            # game.get_ball().is_carried = True 
                    
            
            # xs = [p.position.x for p in ps].append(p_carrier.position.x)
            # ys = [p.position.y for p in ps].append(p_carrier.position.y)
            
            # x_min = min(xs)
            # x_max = max(xs)
            # y_min = min(ys)
            # y_max = max(ys)
            
            
            # for p in home_players: 
                # move_player_out_of_square(game, p, [x_min, y_min], [x_max, y_max])
            
    #        place away team on other side of pitch, of the wings
            # for p in get_away_players(game): 
                # move_player_out_of_square(game, p, [x_min, y_min], [x_max, y_max])        
            
            # self.turn = deepcopy(game.state.home_team.state.turn)  
            
            
        # def training_done(self, game): 
            # training_complete = self.turn  !=  game.state.home_team.state.turn
            
            # num_tackle_zones_at(self, player, position)
            
            # training_outcome = game.state.home_team.state.score > 0 
            # return training_complete, training_outcome
        
        # def allowed_fail_rate(self): 
            # return 0 
    pass
