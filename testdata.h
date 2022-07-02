int const COUNT = 72;
	
float input_s[] = {

	/* A */
	
	0, 0, 0, 1, 1, 0, 0, 0, 
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 1, 0, 0, 0, 0, 1, 0, 
	0, 1, 1, 1, 1, 1, 1, 0,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	
    /* B */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	1, 1, 1, 1, 1, 0, 0, 0,
	
	/* C */
	
	0, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	0, 1, 1, 1, 1, 0, 0, 0, 
	
    /* D */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 0, 1, 0,  
	1, 0, 0, 0, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	1, 1, 1, 1, 1, 0, 0, 0,

	/* E */
	
	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 1, 1, 1, 1, 0, 0, 0, 
	
    /* F */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 

	/* G */
	
	0, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 1, 1, 1, 0, 0,  
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	0, 1, 1, 1, 1, 0, 0, 0, 
	
    /* H */

	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1,

	/* I */
	
	0, 0, 1, 1, 1, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 1, 1, 1, 0, 0, 0, 
	
    /* J */

	0, 0, 1, 1, 1, 1, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0,  
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 1, 0, 0, 1, 0, 0,  
	0, 0, 0, 1, 1, 0, 0, 0, 

	/* K */
	
	1, 0, 0, 0, 1, 0, 0, 0, 
	1, 0, 0, 1, 0, 0, 0, 0, 
	1, 0, 1, 0, 0, 0, 0, 0, 
	1, 1, 0, 0, 0, 0, 0, 0, 
	1, 0, 1, 0, 0, 0, 0, 0,  
	1, 0, 0, 1, 0, 0, 0, 0, 
	1, 0, 0, 0, 1, 0, 0, 0,  
	1, 0, 0, 0, 0, 1, 0, 0,  
	
    /* L */

	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0,  
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0,  
	0, 1, 1, 1, 1, 0, 0, 0, 

	/* M */
	
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 1, 0, 0, 0, 0, 1, 1, 
	1, 0, 1, 0, 0, 1, 0, 1, 
	1, 0, 0, 1, 1, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	
    /* N */

	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 1, 0, 0, 0, 0, 0, 1, 
	1, 0, 1, 0, 0, 0, 0, 1, 
	1, 0, 0, 1, 0, 0, 0, 1, 
	1, 0, 0, 0, 1, 0, 0, 1,  
	1, 0, 0, 0, 0, 1, 0, 1, 
	1, 0, 0, 0, 0, 0, 1, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 

	/* O */
	
	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0, 
	
    /* P */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 

	/* Q */
	
	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 1, 0, 0, 1, 
	1, 0, 0, 0, 0, 1, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,  
	
    /* R */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 1, 0, 0, 0, 0, 0,  
	1, 0, 0, 1, 0, 0, 0, 0, 
	1, 0, 0, 0, 1, 0, 0, 0,  
	1, 0, 0, 0, 0, 1, 0, 0, 

	/* S */
	
	0, 0, 1, 1, 1, 1, 1, 0, 
	0, 1, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 1, 1, 1, 1, 0,  
	0, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0, 
	
    /* T */

	1, 1, 1, 1, 1, 1, 1, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 

	/* U */
	
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0, 
	
    /* V */

	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 0, 0, 0, 0, 1, 0, 
	0, 1, 0, 0, 0, 0, 1, 0, 
	0, 0, 1, 0, 0, 1, 0, 0,  
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 0, 0, 1, 1, 0, 0, 0,  
	0, 0, 0, 1, 1, 0, 0, 0, 

	/* W */
	
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 1, 1, 0, 0, 1,  
	1, 0, 0, 1, 1, 0, 0, 1, 
	1, 0, 1, 0, 0, 1, 0, 1,  
	1, 1, 0, 0, 0, 0, 1, 1, 
	
    /* X */

	1, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 0, 0, 0, 0, 1, 0, 
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 0, 0, 1, 1, 0, 0, 0, 
	0, 0, 0, 1, 1, 0, 0, 0,  
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 1, 0, 0, 0, 0, 1, 0,  
	1, 0, 0, 0, 0, 0, 0, 1, 

	/* Y */
	
	1, 0, 0, 0, 0, 0, 1, 0, 
	0, 1, 0, 0, 0, 1, 0, 0, 
	0, 0, 1, 0, 1, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	
    /* Z */

	1, 1, 1, 1, 1, 1, 1, 1, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0,  
	1, 1, 1, 1, 1, 1, 1, 1,

	/* 0 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 1, 0, 1, 
	1, 0, 0, 0, 1, 0, 0, 1, 
	1, 0, 0, 0, 1, 0, 0, 1, 
	1, 0, 0, 1, 0, 0, 0, 1,  
	1, 0, 0, 1, 0, 0, 0, 1, 
	1, 0, 1, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,
	
	/* 1 */

	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 1, 1, 0, 0, 0, 
	0, 0, 1, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0,  
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0,  
	0, 0, 1, 1, 1, 1, 1, 0,

	/* 2 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0,  
	0, 1, 1, 1, 1, 1, 1, 1,
		
	/* 3 */

	0, 0, 1, 1, 1, 1, 1, 0, 
	0, 1, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 1, 1, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 1,  
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 0, 0, 0, 0, 0, 1,  
	0, 0, 1, 1, 1, 1, 1, 0,
		
	/* 4 */

	0, 0, 1, 0, 0, 0, 0, 1, 
	0, 0, 1, 0, 0, 0, 0, 1, 
	0, 1, 0, 0, 0, 0, 0, 1, 
	0, 1, 1, 1, 1, 1, 1, 1, 
	0, 0, 0, 0, 0, 0, 0, 1,  
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 1,  
	0, 0, 0, 0, 0, 0, 0, 1,
		
	/* 5 */

	1, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 1, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 0,  
	0, 0, 0, 0, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 0, 1, 0,  
	0, 1, 1, 1, 1, 1, 0, 0,
		
	/* 6 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 1, 1, 1, 0,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,
		
	/* 7 */

	1, 1, 1, 1, 1, 1, 1, 1, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0,  
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0,  
	0, 0, 0, 0, 0, 0, 1, 0,

	/* 8 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,

	/* 9 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 1, 1, 1, 1, 1, 1, 
	0, 0, 0, 0, 0, 0, 0, 1,  
	0, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,

		/* A */
	
	0, 0, 0, 1, 1, 0, 0, 0, 
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 1, 0, 0, 0, 0, 1, 0, 
	0, 1, 1, 1, 1, 1, 1, 0,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	
    /* B */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	1, 1, 1, 1, 1, 0, 0, 0,
	
	/* C */
	
	0, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	0, 1, 1, 1, 1, 0, 0, 0, 
	
    /* D */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 0, 1, 0,  
	1, 0, 0, 0, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	1, 1, 1, 1, 1, 0, 0, 0,

	/* E */
	
	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 1, 1, 1, 1, 0, 0, 0, 
	
    /* F */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 

	/* G */
	
	0, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 1, 1, 1, 0, 0,  
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0,  
	0, 1, 1, 1, 1, 0, 0, 0, 
	
    /* H */

	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1,

	/* I */
	
	0, 0, 1, 1, 1, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 1, 1, 1, 0, 0, 0, 
	
    /* J */

	0, 0, 1, 1, 1, 1, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0,  
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 1, 0, 0, 1, 0, 0,  
	0, 0, 0, 1, 1, 0, 0, 0, 

	/* K */
	
	1, 0, 0, 0, 1, 0, 0, 0, 
	1, 0, 0, 1, 0, 0, 0, 0, 
	1, 0, 1, 0, 0, 0, 0, 0, 
	1, 1, 0, 0, 0, 0, 0, 0, 
	1, 0, 1, 0, 0, 0, 0, 0,  
	1, 0, 0, 1, 0, 0, 0, 0, 
	1, 0, 0, 0, 1, 0, 0, 0,  
	1, 0, 0, 0, 0, 1, 0, 0,  
	
    /* L */

	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0,  
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0,  
	0, 1, 1, 1, 1, 0, 0, 0, 

	/* M */
	
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 1, 0, 0, 0, 0, 1, 1, 
	1, 0, 1, 0, 0, 1, 0, 1, 
	1, 0, 0, 1, 1, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	
    /* N */

	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 1, 0, 0, 0, 0, 0, 1, 
	1, 0, 1, 0, 0, 0, 0, 1, 
	1, 0, 0, 1, 0, 0, 0, 1, 
	1, 0, 0, 0, 1, 0, 0, 1,  
	1, 0, 0, 0, 0, 1, 0, 1, 
	1, 0, 0, 0, 0, 0, 1, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 

	/* O */
	
	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0, 
	
    /* P */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	1, 0, 0, 0, 0, 0, 0, 0, 

	/* Q */
	
	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 1, 0, 0, 1, 
	1, 0, 0, 0, 0, 1, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,  
	
    /* R */

	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 0, 0, 0, 0, 1, 0, 0, 
	1, 1, 1, 1, 1, 0, 0, 0, 
	1, 0, 1, 0, 0, 0, 0, 0,  
	1, 0, 0, 1, 0, 0, 0, 0, 
	1, 0, 0, 0, 1, 0, 0, 0,  
	1, 0, 0, 0, 0, 1, 0, 0, 

	/* S */
	
	0, 0, 1, 1, 1, 1, 1, 0, 
	0, 1, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 1, 1, 1, 1, 0,  
	0, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0, 
	
    /* T */

	1, 1, 1, 1, 1, 1, 1, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 

	/* U */
	
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0, 
	
    /* V */

	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 0, 0, 0, 0, 1, 0, 
	0, 1, 0, 0, 0, 0, 1, 0, 
	0, 0, 1, 0, 0, 1, 0, 0,  
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 0, 0, 1, 1, 0, 0, 0,  
	0, 0, 0, 1, 1, 0, 0, 0, 

	/* W */
	
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 1, 1, 0, 0, 1,  
	1, 0, 0, 1, 1, 0, 0, 1, 
	1, 0, 1, 0, 0, 1, 0, 1,  
	1, 1, 0, 0, 0, 0, 1, 1, 
	
    /* X */

	1, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 0, 0, 0, 0, 1, 0, 
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 0, 0, 1, 1, 0, 0, 0, 
	0, 0, 0, 1, 1, 0, 0, 0,  
	0, 0, 1, 0, 0, 1, 0, 0, 
	0, 1, 0, 0, 0, 0, 1, 0,  
	1, 0, 0, 0, 0, 0, 0, 1, 

	/* Y */
	
	1, 0, 0, 0, 0, 0, 1, 0, 
	0, 1, 0, 0, 0, 1, 0, 0, 
	0, 0, 1, 0, 1, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	
    /* Z */

	1, 1, 1, 1, 1, 1, 1, 1, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0,  
	1, 1, 1, 1, 1, 1, 1, 1,

	/* 0 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 1, 0, 1, 
	1, 0, 0, 0, 1, 0, 0, 1, 
	1, 0, 0, 0, 1, 0, 0, 1, 
	1, 0, 0, 1, 0, 0, 0, 1,  
	1, 0, 0, 1, 0, 0, 0, 1, 
	1, 0, 1, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,
	
	/* 1 */

	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 1, 1, 0, 0, 0, 
	0, 0, 1, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0,  
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0,  
	0, 0, 1, 1, 1, 1, 1, 0,

	/* 2 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0,  
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0,  
	0, 1, 1, 1, 1, 1, 1, 1,
		
	/* 3 */

	0, 0, 1, 1, 1, 1, 1, 0, 
	0, 1, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 1, 1, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 1,  
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 0, 0, 0, 0, 0, 1,  
	0, 0, 1, 1, 1, 1, 1, 0,
		
	/* 4 */

	0, 0, 1, 0, 0, 0, 0, 1, 
	0, 0, 1, 0, 0, 0, 0, 1, 
	0, 1, 0, 0, 0, 0, 0, 1, 
	0, 1, 1, 1, 1, 1, 1, 1, 
	0, 0, 0, 0, 0, 0, 0, 1,  
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 1,  
	0, 0, 0, 0, 0, 0, 0, 1,
		
	/* 5 */

	1, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 1, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 0,  
	0, 0, 0, 0, 0, 0, 1, 0, 
	1, 0, 0, 0, 0, 0, 1, 0,  
	0, 1, 1, 1, 1, 1, 0, 0,
		
	/* 6 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 1, 1, 1, 0,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,
		
	/* 7 */

	1, 1, 1, 1, 1, 1, 1, 1, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0,  
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 1, 0,  
	0, 0, 0, 0, 0, 0, 1, 0,

	/* 8 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0,

	/* 9 */

	0, 1, 1, 1, 1, 1, 1, 0, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1, 
	0, 1, 1, 1, 1, 1, 1, 1, 
	0, 0, 0, 0, 0, 0, 0, 1,  
	0, 0, 0, 0, 0, 0, 0, 1, 
	1, 0, 0, 0, 0, 0, 0, 1,  
	0, 1, 1, 1, 1, 1, 1, 0

};

float input_e[] = {

	/* A */
	
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* B */

	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* C */
	
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* D */

	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* E */
	
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* F */

	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* G */
	
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* H */

	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* I */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* J */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* K */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* L */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* M */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* N */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* O */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* P */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* Q */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* R */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* S */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* T */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* U */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* V */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* W */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* X */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* Y */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* Z */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 0 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 1 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 2 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 3 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 4 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 5 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 6 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,	

	/* 7 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,	

	/* 8 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 9 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

		/* A */
	
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* B */

	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* C */
	
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* D */

	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* E */
	
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* F */

	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* G */
	
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* H */

	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* I */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* J */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* K */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* L */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* M */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* N */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* O */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* P */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* Q */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* R */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* S */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* T */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* U */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* V */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* W */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* X */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* Y */
	
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	
    /* Z */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 0 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 1 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 2 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 3 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 4 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 5 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 6 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,	

	/* 7 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,	

	/* 8 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0,

	/* 9 */

	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,  
	0, 0, 0, 0, 0, 0, 0, 0		

};