from mcp.server.fastmcp import FastMCP
from random import choice
from typing import List,AnyStr
mcp = FastMCP(name="Random Name",debug=True,log_level="DEBUG")

@mcp.tool(name="get-random-names",structured_output=True)
def get_random_name(names:List[AnyStr]=None)->AnyStr: # type: ignore
    """
    Get the random peoples name. The names are stored in a local array
    args:
        names: the user can pass in a list of names to choose from, or it will default predefined list names
    """
    if names is None:
        predefined_list = ["Mariappan","Rajalakshmi","Muthukamalan","Revathi","Rajapaul"]
        return choice(predefined_list)
    else:
        return choice(names)



if __name__ == "__main__":
    mcp.run()
